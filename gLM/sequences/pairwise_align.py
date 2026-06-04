import parasail

_matrix = parasail.blosum62


def align_pair(seq1, seq2):
    result = parasail.nw_trace_scan_16(seq1, seq2, 10, 1, _matrix)
    tb = result.traceback
    a1 = tb.query   # contains '-' for gaps
    a2 = tb.ref     # contains '-' for gaps
    return a1, a2
