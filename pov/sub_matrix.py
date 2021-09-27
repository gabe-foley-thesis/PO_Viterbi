from Bio.SubsMat import MatrixInfo, SeqMat

blosum62 = MatrixInfo.blosum62
blosum50 = MatrixInfo.blosum50
from pov.custom_sub_matrix import blosum62LatestProbs_dict, blosum62EstimatedWithX_dict


def score_match(pair, matrix):
    if pair not in matrix:
        return matrix[(tuple(reversed(pair)))]
    else:
        return matrix[pair]


borodovsky_data = {
    ("A", "A"): 0.5,
    ("A", "C"): 0.15,
    ("A", "G"): 0.05,
    ("A", "T"): 0.3,
    ("C", "C"): 0.5,
    ("C", "G"): 0.3,
    ("C", "T"): 0.05,
    ("G", "G"): 0.5,
    ("G", "T"): 0.15,
    ("T", "T"): 0.5,
}


borodovsky_4_7 = SeqMat(borodovsky_data, mat_name="borodovsky_4_7")

blosum62LatestProbs = SeqMat(blosum62LatestProbs_dict, mat_name="blosum62LatestProbs")

blosum62EstimatedWithX = SeqMat(
    blosum62EstimatedWithX_dict, mat_name="blosum62EstimatedWithX"
)
