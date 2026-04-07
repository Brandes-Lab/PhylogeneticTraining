"""
PrefixLM Trainer for ModernBERT.

Subclasses HuggingFace Trainer to override compute_loss with the custom
prefixlm_forward_flash that bypasses ModernBertModel.forward() and injects
the PrefixLM attention mask directly into the encoder layers.

IMPORTANT: Set remove_unused_columns=False in TrainingArguments,
otherwise the Trainer strips prefix_lengths from the batch.
"""

import torch
import random
import numpy as np
from transformers import Trainer
from torch.utils.data import DataLoader

from gLM.models.protein_modernbert_phylo import prefixlm_forward_flash


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class PrefixLMTrainer(Trainer):
    """
    HuggingFace Trainer subclass for PrefixLM training.

    Overrides compute_loss to use prefixlm_forward_flash instead of the default
    model(**inputs), which would call ModernBertModel.forward() and crash
    because _update_attention_mask cannot handle our custom 4D mask.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, logits = prefixlm_forward_flash(model, inputs, model.device)
        return (loss, {"logits": logits}) if return_outputs else loss

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers if self.args.dataloader_num_workers > 0 else False,
            prefetch_factor=self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers > 0 else None,
            worker_init_fn=_seed_worker,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("No eval dataset provided.")
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=self.data_collator,
            drop_last=False,
        )