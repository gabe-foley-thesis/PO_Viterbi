import pytest
import pov.pair_hmm as ph
import pov.alignment_profile as aln_profile
import pov.sub_matrix as sub_matrix
import pov.align as align
import pov.parameters as params
import pov.sequence as sequence

@pytest.fixture
def simple_blosum_62_2():
    seqs = sequence.readFastaFile("./files/simple_seqs/simple_2.fasta")

    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.basic_params, [profile1, profile2], sub_matrix.blosum62, log_transform=False)

    return phmm

@pytest.fixture
def probabilities_blosum_62_2():
    seqs = sequence.readFastaFile("./files/simple_seqs/simple_2.fasta")

    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.basic_params, [profile1, profile2], sub_matrix.blosum62LatestProbs,
                             log_transform=False)

    return phmm

@pytest.fixture
def simple_blosum_50_2():
    seqs = sequence.readFastaFile("./files/simple_seqs/simple_2.fasta")

    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.basic_params, [profile1, profile2], sub_matrix.blosum50, log_transform=False)

    return phmm

@pytest.fixture
def durbin_blosum_50_2():

    seqs = sequence.readFastaFile("./files/simple_seqs/durbin_2.fasta")
    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.basic_params, [profile1, profile2], sub_matrix.blosum50, log_transform=False)

    return phmm


@pytest.fixture
def borodovsky_blosum_50_2():

    seqs = sequence.readFastaFile("./files/simple_seqs/borodovsky.fasta")
    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.borodovsky_4_7, [profile1, profile2], sub_matrix.blosum62LatestProbs,
                             log_transform=False)

    return phmm

@pytest.fixture
def borodovsky_blosum_50_2():

    seqs = sequence.readFastaFile("./files/simple_seqs/borodovsky.fasta")
    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.borodovsky_4_7, [profile1, profile2], sub_matrix.blosum62LatestProbs,
                             log_transform=False)

    return phmm

@pytest.fixture
def ox_104t17_1():

    seqs = sequence.readFastaFile("./files/qscore_corrections/ox_104t17_1.fasta")
    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.qscore_params, [profile1, profile2], sub_matrix.blosum62EstimatedWithX,
                             log_transform=True)

    return phmm


# Custom Alignments
@pytest.fixture
def monkey_62_2():

    seqs = sequence.readFastaFile("./files/custom_seqs/monkey.fasta")
    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.basic_params, [profile1, profile2], sub_matrix.blosum62, log_transform=False)


    return phmm

# Custom Alignments
@pytest.fixture
def single_col_62_2():

    seqs = sequence.readFastaFile("./files/custom_seqs/single_col.fasta")
    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.basic_params, [profile1, profile2], sub_matrix.blosum62, log_transform=False)


    return phmm

# Custom Alignments
@pytest.fixture
def two_col_62_2():

    seqs = sequence.readFastaFile("./files/custom_seqs/2_col.fasta")
    profile1 = aln_profile.AlignmentProfile([seqs[0]])
    profile2 = aln_profile.AlignmentProfile([seqs[1]])

    phmm = align.load_params(params.basic_params, [profile1, profile2], sub_matrix.blosum62, log_transform=False)


    return phmm