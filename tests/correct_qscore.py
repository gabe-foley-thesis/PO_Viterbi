import pytest
from pov.align import align_seqs
import pov.pair_hmm  as ph
import math
import numpy as np
from pandas import DataFrame

def compare_matrices(query, correct, check_log=True):
    for i in range(query.shape[0]):
        for j in range(query.shape[1]):
            # print (i,j)
            # print (query[i][j])
            # print (correct[i][j])
            # print()
            if (query[i][j] == 0.0 or query[i][j] == float("-Inf")) and (correct[i][j] == 0.0 or correct[i][j] ==
                float(\
                    "-Inf")):
                # print ('it was')
                continue



            elif check_log and math.isclose(math.log(query[i][j]), math.log(correct[i][j]) ,rel_tol = 1e-4):
                continue

            elif not check_log and math.isclose(query[i][j], correct[i][j] ,rel_tol = 1e-2):
                continue
            else:
                if check_log:
                    return f'False - at {i} and {j} and {math.log(query[i][j])} is not the same as {math.log(correct[i][j])}'
                else:
                    return f'False - at {i} and {j} and {query[i][j]} is not the same as {correct[i][j]}'

    return True


def test_ox_104t17_1(ox_104t17_1):

    ox_104t17_1.performMEAAlignment()

    aligned_profile = ox_104t17_1.get_alignment('mea')

    print (aligned_profile)

    assert "".join(aligned_profile.seqs[0].sequence) == \
           "IPAYLAETLYYAMKGAGTDDHTLIRVMVSRSEIDLFNIRKEFRKNFATSLYSMIKGDTSGDYKKALLLLC---"
    assert "".join(aligned_profile.seqs[1].sequence) == "IPAYLAETLYYAMKGAGTDDHTLIRVIVSRSEIDLFNIRKEFRKNFATSLYSMIKGDTSGDYKKALLLLCGGE"
