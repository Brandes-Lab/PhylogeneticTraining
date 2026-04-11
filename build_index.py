from Bio import SeqIO
import time

start = time.time()
index = SeqIO.index_db(
    "/gpfs/data/brandeslab/rm7569/uniref100.idx",
    ["/gpfs/data/brandeslab/Data/uniref/uniref100.fasta"],
    "fasta"
)
print(f"Done. {len(index)} records in {(time.time()-start)/60:.1f} min.")