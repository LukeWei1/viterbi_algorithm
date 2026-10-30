#!/usr/bin/env python3
"""Export the Brown corpus (via NLTK) to data/brown_tagged.txt in word/TAG format.

    pip install nltk
    python3 scripts/export_brown.py              # 12-tag universal tagset (recommended)
    python3 scripts/export_brown.py --full       # original ~470 Brown tags (harder)
"""
import sys
import nltk

nltk.download("brown", quiet=True)
nltk.download("universal_tagset", quiet=True)
from nltk.corpus import brown

full = "--full" in sys.argv
sents = brown.tagged_sents() if full else brown.tagged_sents(tagset="universal")
out = "data/brown_full_tagged.txt" if full else "data/brown_tagged.txt"

with open(out, "w") as f:
    for sent in sents:
        # a word never contains whitespace in Brown; '/' inside words is fine (we split on the LAST '/')
        f.write(" ".join(f"{w}/{t}" for w, t in sent) + "\n")
print(f"wrote {out}: {len(sents)} sentences")
