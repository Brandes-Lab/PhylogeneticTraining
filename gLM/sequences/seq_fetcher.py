from Bio import SeqIO
import os

class SequenceFetcher:
    def __init__(self, fasta_path, index_db_path):
        # Always rebuild index (safe)
        if os.path.exists(index_db_path):
            os.remove(index_db_path)

        self.index = SeqIO.index_db(index_db_path, [fasta_path], "fasta")

    def __call__(self, seq_id):
        record = self.index[seq_id]
        return str(record.seq)