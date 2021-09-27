import pov.alignment_profile as aln_profile
from pov.sequence import Sequence
def test_profile_with_one_sequence():
    profile_1 = aln_profile.AlignmentProfile([Sequence('RTAG')])

    assert profile_1.profile[0]['R'] == 1

def test_profile_with_two_sequences():
    profile_2 = aln_profile.AlignmentProfile([Sequence('RTAG'), Sequence('-TA-')])
    assert profile_2.profile[1]['T'] == 2