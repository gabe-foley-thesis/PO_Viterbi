""" Check for each position in a given ancestor that the presence of ancestral content implied to be there by a given alignment and tree is not substantially less parsimonious then the alternative of not having ancestral content there."""

import argparse
import sys
import pov.align as align
import pov.sub_matrix as sub_matrix

params = {'tau': 0.0002, 'epsilon': 0.0004, 'delta': 0.0002, 'emissionX': 0.2, 'emissionY':
    0.2}

from pov import __version__

def run(args):
    print( "Running Partial Order Viterbi v" + __version__ + "\n")



    aligned_profile = align.align_seqs(args.fasta, args.outputfile, aln_type="viterbi",
                                       params=params, subsmat=sub_matrix.blosum62EstimatedWithX_dict,
                                       log_transform=False)

def pov_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--fasta", help="Path to fasta", required=True)
    parser.add_argument("-o", "--outputfile", help="Path to write out alignment to", required=True)
    return parser


def main(args=None):
    parser = pov_parser()
    if args:
        sys.argv = args
    if len(sys.argv) > 1 and sys.argv[1] in {'-v', '--v', '-version', '--version'}:
        print(f'Partial Order Viterbi v{__version__}')
        sys.exit(0)
    else:
        print(f'Partial Order Viterbi v{__version__}')
        args = parser.parse_args(args)
        run(args)
    sys.exit(0)


if __name__ == "__main__":
    main()