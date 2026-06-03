from Bio import SeqIO
import time

start = time.time()
index = SeqIO.index_db(
    "/gpfs/data/brandeslab/User/as12267/uniref90.idx",
    ["/gpfs/data/brandeslab/Data/uniref/uniref90.fasta"],
    "fasta"
)
print(f"Done. {len(index)} records in {(time.time()-start)/60:.1f} min.")