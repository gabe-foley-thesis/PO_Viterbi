# Partial Order Viterbi

This code runs a Partial Order Alignment variant of the Viterbi algorithm.

It isn't currently in development.


# Installation

1. Clone the repo

```
git clone https://github.com/gabefoley/PO_Viterbi
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Try a test alignment from the tests folder

```
python -m pov/__main__.py -a tests/files/simple_seqs/simple_6.fasta -o ./simple_6.aln
```

This should give you an alignment that looks like this ->


![Workflow](https://raw.githubusercontent.com/gabefoley/PO_Viterbi/main/images/alignment.png)
