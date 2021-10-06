import sequence as sequence
from sym import Alphabet
import alignment_profile as aln_profile
import guide_tree as gt
import pair_hmm as ph
import sub_matrix as sub_matrix
import parameters as parameters

Protein_Alphabet_wB_X_Z = Alphabet("ABCDEFGHIKLMNPQRSTVWYXZ")


# Align sequences
def align_seqs(
    inpath,
    outpath,
    aln_type,
    params=parameters.basic_params,
    subsmat=sub_matrix.blosum62EstimatedWithX_dict,
    log_transform=True,
):

    print("params are")
    print(params)

    # Read sequences in
    seqs = sequence.readFastaFile(inpath, alphabet=Protein_Alphabet_wB_X_Z)

    print(len(seqs))

    if len(seqs) == 2:
        aln_order = [("N0", [seqs[0].name, seqs[1].name])]

    else:

        # Calculate guide tree
        guide_tree = gt.get_guide_tree(seqs, random=False)
        print(guide_tree.ascii_art())

        # Get the alignment order
        aln_order = gt.get_aln_order(guide_tree)
        # print (aln_order)

    print(aln_order)

    seq_dict = {x.name: x for x in seqs}

    # Predecessors start off blank
    predecessors = [{}, {}]

    # Create alignment in order from guide tree
    for node in aln_order:

        # Get the current node name and list of sequences under that node
        curr_node = node[0]
        curr_seqs = node[1]

        # List to store the aligned sequences in
        aligned = []

        # While the node has sequences underneath yet to be aligned
        while curr_seqs:

            # Get a sequence
            seq = curr_seqs.pop()

            # Make it into a profile if it isn't one already
            if type(seq_dict[seq]) != aln_profile.AlignmentProfile:
                profile = aln_profile.AlignmentProfile([seq_dict[seq]])
            else:
                profile = seq_dict[seq]

            # Add sequence to the aligned list
            aligned.append(profile)

            # If we have two profiles it is time to align
            if len(aligned) > 1:

                pair_hmm = load_params(
                    params, aligned, subsmat, log_transform, predecessors
                )

                if aln_type == "viterbi":

                    pair_hmm.performViterbiAlignment(po=False)
                    aligned_profile = pair_hmm.get_alignment(type_to_get="viterbi")

                elif aln_type == "poviterbi":

                    pair_hmm.performViterbiAlignment(po=True)
                    aligned_profile = pair_hmm.get_alignment(type_to_get="viterbi")

                elif aln_type == "mea":

                    pair_hmm.performMEAAlignment(po=False)
                    aligned_profile = pair_hmm.get_alignment(type_to_get="mea")

                elif aln_type == "pomea":

                    pair_hmm.performMEAAlignment(po=True)
                    aligned_profile = pair_hmm.get_alignment(type_to_get="mea")

                # Clear the previous unaligned sequences
                aligned = []

                # Add the aligned sequences
                aligned.append(aligned_profile)

        seq_dict[curr_node] = aligned[0]

    with open(outpath, "w") as outfile:
        outfile.write(str(aligned_profile))

    return aligned_profile


def load_params(params, seqs, subsmat, log_transform, predecessors=[{}, {}]):
    pair_hmm = ph.PairHMM(
        seqs,
        params["tau"],
        params["epsilon"],
        params["delta"],
        params["emissionX"],
        params["emissionY"],
        subsmat,
        log_transform,
    )
    return pair_hmm
