#!/usr/bin/env python3
"""Generate a small SYNTHETIC tagged corpus (data/sample_tagged.txt).

This is only a smoke test so the POS pipeline can be run without downloading
Brown. Its accuracy numbers say nothing about real English.
"""
import random

random.seed(0)
DET = ["the", "a", "this", "that", "every"]
ADJ = ["big", "small", "quick", "lazy", "happy", "red", "old", "bright", "quiet", "famous", "readable", "joyful"]
NOUN = ["dog", "cat", "man", "woman", "city", "book", "river", "teacher", "student", "garden", "idea", "car", "table", "doctor", "bird"]
PROPN = ["Alice", "Bob", "Boston", "Paris", "Newton", "Harvard", "Maria", "Chicago"]
VERB = ["sees", "likes", "finds", "reads", "builds", "visits", "loves", "watches", "follows", "paints"]
VERB_INTR = ["sleeps", "runs", "waits", "arrives", "sings", "smiles"]
ADV = ["quickly", "slowly", "often", "quietly", "happily", "rarely"]
PREP = ["in", "near", "behind", "with", "under"]
# Ambiguous words that can be both noun and verb -> gives the HMM something to resolve
AMB_N = ["watch", "walk", "play", "paint", "run"]
AMB_V = ["watch", "walk", "play", "paint", "run"]


SYL = ["ba", "ko", "ri", "tu", "ve", "lo", "mi", "sa", "dre", "plo", "gri", "zan"]


def rare(kind):
    """A made-up word with a suffix cue, so out-of-vocabulary handling can be tested."""
    stem = "".join(random.choice(SYL) for _ in range(random.choice([2, 3])))
    if kind == "NOUN":
        return stem + random.choice(["tion", "ness", "ment", "ity", "er"])
    if kind == "ADJ":
        return stem + random.choice(["ous", "al", "ful", "able"])
    if kind == "ADV":
        return stem + "ly"
    if kind == "PROPN":
        return stem.capitalize()
    return stem + random.choice(["ed", "ing", "s"])  # verb-ish


def maybe_rare(tok):
    """Replace a token by a rare made-up word of the same tag with small probability."""
    w, t = tok
    if t in ("NOUN", "ADJ", "ADV", "VERB") and random.random() < 0.07:
        kind = "PROPN" if (t == "NOUN" and w[0].isupper()) else t
        return (rare(kind), t)
    return tok


def np():
    toks = []
    if random.random() < 0.15:
        return [(random.choice(PROPN), "NOUN")]
    toks.append((random.choice(DET), "DET"))
    for _ in range(random.choice([0, 0, 1, 2])):
        toks.append((random.choice(ADJ), "ADJ"))
    pool = NOUN + AMB_N if random.random() < 0.8 else NOUN
    toks.append((random.choice(pool), "NOUN"))
    return toks


def sentence():
    s = np()
    r = random.random()
    if r < 0.5:
        s += [(random.choice(VERB), "VERB")] + np()
    elif r < 0.75:
        s += [(random.choice(VERB_INTR), "VERB")]
    else:
        s += [("will", "VERB"), (random.choice(AMB_V), "VERB")] + np()
    if random.random() < 0.4:
        s.append((random.choice(ADV), "ADV"))
    if random.random() < 0.4:
        s += [(random.choice(PREP), "ADP")] + np()
    s.append((".", "."))
    return [maybe_rare(x) for x in s]


with open("data/sample_tagged.txt", "w") as f:
    for _ in range(3000):
        f.write(" ".join(f"{w}/{t}" for w, t in sentence()) + "\n")
print("wrote data/sample_tagged.txt")
