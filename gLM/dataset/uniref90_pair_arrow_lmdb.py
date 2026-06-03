import os
import csv
import random
import lmdb
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from gLM.sequences.pairwise_align import align_pair, percent_identity



def _open_lmdb(path: str):
    """Open an LMDB environment with settings tuned for multi-worker reads."""
    return lmdb.open(
        path,
        readonly=True,
        lock=False,       # no file lock — essential for multiple readers
        subdir=True,
        readahead=False,  # disable OS readahead for random access patterns
        meminit=False,    # don't zero-init memory pages
        map_size=1024 * 1024 ** 3,  # 1 TB virtual map (no RAM cost until accessed)
    )


class Uniref90ArrowDatasetForLMDB_deterministic(Dataset):
    """
    Deterministic LMDB-backed UniRef90 dataset.

    For a given seed and cluster index, this always selects the same pair:
        same seed + same ridx + same member_ids => same m1, m2

    Optionally logs selected pairs to CSV.
    """

    def __init__(
        self,
        dataset_path: str,
        training_type: str,
        lmdb_path: str,
        max_tries: int = 20,
        seed: int = 42,
        pair_log_csv: str | None = None,
        run_name: str | None = None,
    ):
        super().__init__()
        self.training_type = training_type
        self.dataset_path  = dataset_path
        self.lmdb_path     = lmdb_path
        self.max_tries     = max_tries
        self.seed          = seed
        self.pair_log_csv  = pair_log_csv
        self.run_name      = run_name

        self._env          = None
        self._dataset      = None

    def _get_env(self):
        if self._env is None:
            self._env = _open_lmdb(self.lmdb_path)
        return self._env

    def _get_dataset(self):
        if self._dataset is None:
            self._dataset = load_from_disk(self.dataset_path)
        return self._dataset

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_dataset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self._get_dataset())

    @staticmethod
    def _fetch_seq(txn, member_id: str) -> str:
        v = txn.get(member_id.encode("utf-8"))
        if v is None:
            raise KeyError(member_id)
        return v.decode("ascii", errors="strict")

    def _log_pair(self, idx, ridx, cluster_id, m1, m2):
        if self.pair_log_csv is None:
            return

        rank = int(os.environ.get("RANK", "0"))

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        base, ext = os.path.splitext(self.pair_log_csv)
        if ext == "":
            ext = ".csv"

        out_csv = f"{base}.rank{rank}.worker{worker_id}{ext}"

        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None

        file_exists = os.path.exists(out_csv)

        with open(out_csv, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "run_name",
                    "training_type",
                    "rank",
                    "worker_id",
                    "idx",
                    "ridx",
                    "cluster_id",
                    "member_id_1",
                    "member_id_2",
                ])

            writer.writerow([
                self.run_name,
                self.training_type,
                rank,
                worker_id,
                idx,
                ridx,
                cluster_id,
                m1,
                m2,
            ])

    def __getitem__(self, idx):
        dataset = self._get_dataset()
        n       = len(dataset)
        env     = self._get_env()

        for attempt in range(self.max_tries):
            ridx       = (idx + attempt) % n
            ex         = dataset[ridx]
            member_ids = ex["member_ids"]

            if not member_ids or len(member_ids) < 2:
                continue

            # Deterministic pair for this cluster index.
            rng = random.Random(self.seed + ridx)
            m1, m2 = rng.sample(member_ids, 2)

            cluster_id = ex.get("cluster_id", None)
            self._log_pair(idx, ridx, cluster_id, m1, m2)

            try:
                with env.begin(write=False) as txn:
                    s1 = self._fetch_seq(txn, m1)
                    s2 = self._fetch_seq(txn, m2)
            except (KeyError, lmdb.Error):
                self._env = None
                env = self._get_env()
                continue

            return self._format(s1, s2)

        raise RuntimeError(
            f"Could not find a valid sample after {self.max_tries} tries "
            f"starting from idx={idx}"
        )

    def _format(self, s1: str, s2: str):
        if self.training_type == "phylo_encoder_only":
            a1, a2 = align_pair(s1, s2)
            if len(a1) != len(a2):
                raise ValueError("Aligned lengths differ")
            return (a1, a2, percent_identity(a1, a2))

        elif self.training_type == "phylo_encoder_decoder":
            return (s1, s2)

        else:
            raise ValueError(
                f"Uniref90ArrowDatasetForLMDB_deterministic only supports "
                f"phylo_encoder_only and phylo_encoder_decoder, got: {self.training_type}"
            )

class Uniref90ArrowDatasetForLMDB(Dataset):
    """
    Training dataset. Each __getitem__ call:
      - Uses the provided idx to walk forward on bad samples — no recursion.
      - LMDB env and HF dataset are both opened lazily per worker process.

    Multi-worker correctness:
      - __getstate__ discards open handles so pickling to workers is safe.
      - worker_init_fn (above) resets handles each epoch for persistent workers.
    """

    def __init__(self, dataset_path: str, training_type: str, lmdb_path: str,
                 max_tries: int = 20):
        super().__init__()
        self.training_type = training_type
        self.dataset_path  = dataset_path
        self.lmdb_path     = lmdb_path
        self.max_tries     = max_tries
        self._env          = None
        self._dataset      = None

    def _get_env(self):
        if self._env is None:
            self._env = _open_lmdb(self.lmdb_path)
        return self._env

    def _get_dataset(self):
        if self._dataset is None:
            self._dataset = load_from_disk(self.dataset_path)
        return self._dataset

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"]     = None
        state["_dataset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self._get_dataset())

    @staticmethod
    def _fetch_seq(txn, member_id: str) -> str:
        v = txn.get(member_id.encode("utf-8"))
        if v is None:
            raise KeyError(member_id)
        return v.decode("ascii", errors="strict")

    def __getitem__(self, idx):
        dataset = self._get_dataset()
        n       = len(dataset)
        env     = self._get_env()

        for attempt in range(self.max_tries):
            ridx       = (idx + attempt) % n
            ex         = dataset[ridx]
            member_ids = ex["member_ids"]

            if not member_ids or len(member_ids) < 2:
                continue

            m1, m2 = random.sample(member_ids, 2)

            try:
                with env.begin(write=False) as txn:
                    s1 = self._fetch_seq(txn, m1)
                    s2 = self._fetch_seq(txn, m2)
            except (KeyError, lmdb.Error):
                # lmdb.Error covers the closed-env case — reset and reopen
                self._env = None
                env = self._get_env()
                continue

            return self._format(s1, s2)

        raise RuntimeError(
            f"Could not find a valid sample after {self.max_tries} tries "
            f"starting from idx={idx}"
        )

    def _format(self, s1: str, s2: str):
        if self.training_type == "MLM":
            return s1 if random.random() < 0.5 else s2

        elif self.training_type == "phylo_encoder_only":
            a1, a2 = align_pair(s1, s2)
            if len(a1) != len(a2):
                raise ValueError("Aligned lengths differ")
            # return (a1, a2, percent_identity(a1, a2))
            return (a1, a2)

        elif self.training_type == "phylo_encoder_decoder":
            return (s1, s2)

        else:
            raise ValueError(f"Unknown training_type: {self.training_type}")


# =============================================================================
# Deterministic eval dataset
# =============================================================================

class Uniref90ArrowEvalDatasetForLMDB(Dataset):
    """
    Evaluation dataset. Uses idx directly (deterministic).
    Same lazy-open + pickle + worker_init_fn pattern as training dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        training_type: str,
        lmdb_path: str,
        max_tries: int = 30,
        deterministic_pair: bool = True,
    ):
        super().__init__()
        self.training_type      = training_type
        self.dataset_path       = dataset_path
        self.lmdb_path          = lmdb_path
        self.max_tries          = max_tries
        self.deterministic_pair = deterministic_pair
        self._env               = None
        self._dataset           = None

    def _get_env(self):
        if self._env is None:
            self._env = _open_lmdb(self.lmdb_path)
        return self._env

    def _get_dataset(self):
        if self._dataset is None:
            self._dataset = load_from_disk(self.dataset_path)
        return self._dataset

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"]     = None
        state["_dataset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self._get_dataset())

    @staticmethod
    def _fetch_seq(txn, member_id: str) -> str:
        v = txn.get(member_id.encode("utf-8"))
        if v is None:
            raise KeyError(member_id)
        return v.decode("ascii", errors="strict")

    def __getitem__(self, idx):
        dataset    = self._get_dataset()
        ex         = dataset[idx]
        member_ids = ex["member_ids"]

        if not member_ids or len(member_ids) < 2:
            raise RuntimeError(f"Cluster at idx={idx} has fewer than 2 members")

        env = self._get_env()

        if self.deterministic_pair:
            m1, m2 = member_ids[0], member_ids[1]
            try:
                with env.begin(write=False) as txn:
                    s1 = self._fetch_seq(txn, m1)
                    s2 = self._fetch_seq(txn, m2)
            except lmdb.Error as e:
                raise RuntimeError(
                    f"LMDB error for deterministic pair ({m1}, {m2}) at idx={idx}"
                ) from e
            return self._format(s1, s2)

        else:
            for _ in range(self.max_tries):
                m1, m2 = random.sample(member_ids, 2)
                try:
                    with env.begin(write=False) as txn:
                        s1 = self._fetch_seq(txn, m1)
                        s2 = self._fetch_seq(txn, m2)
                except (KeyError, lmdb.Error):
                    self._env = None
                    env = self._get_env()
                    continue
                return self._format(s1, s2)

            raise RuntimeError(
                f"Failed to get a valid eval sample for idx={idx} "
                f"after {self.max_tries} tries"
            )

    def _format(self, s1: str, s2: str):
        if self.training_type == "MLM":
            return s1

        elif self.training_type == "phylo_encoder_only":
            a1, a2 = align_pair(s1, s2)
            if len(a1) != len(a2):
                raise RuntimeError("Aligned lengths differ")
            return (a1, a2, percent_identity(a1, a2))

        elif self.training_type == "phylo_encoder_decoder":
            return (s1, s2)

        else:
            raise ValueError(f"Unknown training_type: {self.training_type}")