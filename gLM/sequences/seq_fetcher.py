import os
from Bio import SeqIO


class SequenceFetcher:
    def __init__(self, fasta_path, index_db_path):
        # Store paths only. Don't open the SQLite index here — if we did,
        # forked DataLoader workers would inherit the same connection and
        # clobber each other's reads.
        self.fasta_path = fasta_path
        self.index_db_path = index_db_path

        # Lazy-init: track the open index and the PID it was opened in.
        self._index = None
        self._pid = None

    def _ensure_index(self):
        # Open (or reopen) if never opened, or if PID has changed (i.e. we
        # forked and inherited a parent's connection). Each worker ends up
        # with its own private connection.
        if self._index is None or self._pid != os.getpid():
            self._index = SeqIO.index_db(
                self.index_db_path, [self.fasta_path], "fasta"
            )
            self._pid = os.getpid()
        return self._index

    def __call__(self, seq_id):
        # PID check runs every call — cheap, and the index is cached after
        # the first call in each worker.
        return str(self._ensure_index()[seq_id].seq)