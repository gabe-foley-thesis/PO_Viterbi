import pytest
import pov.pair_hmm as pair_hmm

def test_transition_calculations(simple_blosum_62_2):


    assert simple_blosum_62_2.transitionMM == 0.939
    assert simple_blosum_62_2.transitionMX == 0.03
    assert simple_blosum_62_2.transitionMY == 0.03






