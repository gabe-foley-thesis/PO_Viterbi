import pov.align as align
import pytest
import os

def test_reading_file_and_writing_works():
    align.align_seqs('./files/simple_seqs/simple_4.fasta', './files/simple_seqs/simple_4.aln', aln_type='viterbi')
    assert(os.path.exists('./files/simple_seqs/simple_4.aln'))
    os.remove('./files/simple_seqs/simple_4.aln')

