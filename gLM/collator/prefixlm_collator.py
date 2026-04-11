"""
PrefixLM Data Collator for paired protein sequences.
Packs each (seq1, seq2) pair as: [CLS] seq2 [SEP] seq1 [SEP]
Returns input_ids, labels (with -100 on prefix), and prefix_lengths.
The attention mask is NOT built here — it is constructed inside the model's
forward pass using prefix_lengths, because the mask needs the model's dtype
and the layer-by-layer bypass requires it at forward time.
"""
import torch


class PrefixLMCollator:
    """
    Collator for PrefixLM training: P(seq1 | seq2).
    Packs each pair as: [CLS] seq2_tokens [SEP] seq1_tokens [SEP]
    Labels: -100 on prefix positions + final [SEP] + padding,
            real token IDs on suffix positions.
    After the autoregressive shift in the model forward:
        logits[SEP_position] predicts first seq1 token
        logits[seq1_i]       predicts seq1_{i+1}
    Args:
        tokenizer: tokenizer with cls_token_id, sep_token_id, pad_token_id
        max_seq_len: maximum total packed sequence length (default 4096)
        has_pid: if True, batch items are (seq1, seq2, percent_identity) triples
                 if False, batch items are (seq1, seq2) pairs
    """

    def __init__(self, tokenizer, max_seq_len: int = 4096, has_pid: bool = True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.has_pid = has_pid

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples from dataset.__getitem__
                   Either (seq1, seq2) or (seq1, seq2, percent_identity)
        """
        if self.has_pid:
            s1s, s2s, pids = zip(*batch)
        else:
            s1s, s2s = zip(*batch)
            pids = None

        # Tokenize without special tokens — we add [CLS]/[SEP] manually
        enc_s2 = self.tokenizer(list(s2s), add_special_tokens=False)["input_ids"]
        enc_s1 = self.tokenizer(list(s1s), add_special_tokens=False)["input_ids"]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        all_input_ids = []
        all_prefix_lengths = []

        for s2_ids, s1_ids in zip(enc_s2, enc_s1):
            # Pack: [CLS] seq2 [SEP] seq1 [SEP]
            packed = [cls_id] + s2_ids + [sep_id] + s1_ids + [sep_id]

            # Truncate both sequences proportionally if too long
            if len(packed) > self.max_seq_len:
                # Available tokens for both sequences (minus 3 special tokens: [CLS], [SEP], [SEP])
                available = self.max_seq_len - 3
                total = len(s2_ids) + len(s1_ids)
                # Split available tokens proportionally to original lengths
                s2_keep = int(available * len(s2_ids) / total)
                s1_keep = available - s2_keep
                s2_ids = s2_ids[:s2_keep]
                s1_ids = s1_ids[:s1_keep]
                packed = [cls_id] + s2_ids + [sep_id] + s1_ids + [sep_id]

            # prefix_len recalculated after truncation so it's always correct
            prefix_len = 1 + len(s2_ids) + 1  # [CLS] + seq2 + [SEP]
            all_input_ids.append(packed)
            all_prefix_lengths.append(prefix_len)

        # Pad to longest in batch
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        labels_list = []

        for input_ids, prefix_len in zip(all_input_ids, all_prefix_lengths):
            pad_len = max_len - len(input_ids)

            # Labels: -100 on prefix, real IDs on suffix, -100 on final [SEP] + padding
            labels = list(input_ids)
            for i in range(prefix_len):
                labels[i] = -100
            labels[-1] = -100  # final [SEP]

            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            labels_list.append(labels + [-100] * pad_len)

        batch_out = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "prefix_lengths": torch.tensor(all_prefix_lengths, dtype=torch.long),
        }

        if pids is not None:
            batch_out["percent_identity"] = torch.tensor(pids, dtype=torch.float32)

        return batch_out