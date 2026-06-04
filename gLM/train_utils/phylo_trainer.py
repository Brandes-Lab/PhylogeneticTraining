from transformers import Trainer
from torch.utils.data import DataLoader
import torch
import random
import numpy as np


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class PhyloTrainer(Trainer):
    def get_train_dataloader(self):
        # Per-rank generator so each GPU shuffles independently and its
        # dataloader workers derive distinct Python RNG seeds. Without this
        # offset, every rank would draw the same cluster indices and the
        # same (m1, m2) pairs, making DDP all-reduce average identical
        # gradients across ranks.
        generator = torch.Generator()
        generator.manual_seed(self.args.seed + self.args.process_index)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            worker_init_fn=seed_worker,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )


