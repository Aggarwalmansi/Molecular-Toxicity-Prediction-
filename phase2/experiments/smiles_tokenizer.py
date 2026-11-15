"""
smiles_tokenizer.py
===================
Turns a SMILES string into a list of integers so a Transformer can read it.

Why we need this. The ANN branch reads a fingerprint, which is already a
vector of numbers. The Transformer branch reads the SMILES text instead, and a
neural network cannot read text directly. So we split the text into small
pieces ("tokens") and give each distinct piece its own number.

Why we do not split on single characters. In SMILES, "Cl" is one chlorine
atom, not a carbon followed by an L. "[nH]" is one atom in brackets. The regular
expression below keeps those groups together.

Leakage rule: the vocabulary is built from TRAINING molecules only. A token
that appears for the first time in validation or test becomes <unk>.
"""

import re

# Order matters: longer patterns must come first so "Cl" is matched before "C".
TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]"          # anything inside square brackets, e.g. [nH], [Na+]
    r"|Br|Cl"               # two-letter atoms
    r"|[BCNOSPFIbcnosp]"    # one-letter atoms (lower case = aromatic)
    r"|%[0-9]{2}"           # ring closures above 9, e.g. %10
    r"|[0-9]"               # ring closure digits
    r"|[()=#@+\-\\/:~*$.])" # bonds, branches, charges, stereochemistry
)

PAD_TOKEN = "<pad>"   # filler so every molecule in a batch has the same length
UNK_TOKEN = "<unk>"   # any token the training set never showed us
PAD_ID = 0
UNK_ID = 1


def split_smiles(smiles):
    """Splits one SMILES string into a list of token strings."""
    return TOKEN_PATTERN.findall(smiles)


def build_vocabulary(training_smiles):
    """
    Builds the token -> integer dictionary from training molecules only.

    Returns a dictionary. <pad> is always 0 and <unk> is always 1, so the
    embedding layer can use padding_idx=0.
    """
    tokens = set()
    for smiles in training_smiles:
        tokens.update(split_smiles(smiles))

    vocabulary = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token in sorted(tokens):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode(smiles, vocabulary, max_length):
    """
    Converts one SMILES string into a fixed-length list of integers.

    Molecules shorter than max_length are padded with zeros on the right.
    Molecules longer than max_length are cut, which affects only a small
    number of very large molecules.
    """
    ids = [vocabulary.get(token, UNK_ID) for token in split_smiles(smiles)]
    ids = ids[:max_length]
    ids = ids + [PAD_ID] * (max_length - len(ids))
    return ids


def encode_many(smiles_list, vocabulary, max_length):
    """Encodes a list of SMILES strings into a list of equal-length lists."""
    return [encode(smiles, vocabulary, max_length) for smiles in smiles_list]
