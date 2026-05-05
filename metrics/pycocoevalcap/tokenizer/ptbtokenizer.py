#!/usr/bin/env python
#
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# and refactored to Python 3.
# Image-specific names and comments have been changed to be audio-specific
# Java dependency removed: tokenization is now done in pure Python.
# =================================================================

import re

# punctuations to be removed from the sentences
PUNCTUATIONS = {"''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"}

def _python_tokenize(sentence):
    """Lowercase and remove punctuation tokens, mimicking PTB tokenizer output."""
    sentence = sentence.lower()
    sentence = re.sub(r"([.?!,;:\-])", r" \1 ", sentence)
    tokens = sentence.split()
    tokens = [w for w in tokens if w not in PUNCTUATIONS]
    return ' '.join(tokens)

class PTBTokenizer:
    """Pure-Python replacement for the Stanford PTBTokenizer (no Java required)."""

    def tokenize(self, captions_for_audio):
        final_tokenized_captions_for_audio = {}
        for k, captions in captions_for_audio.items():
            final_tokenized_captions_for_audio[k] = []
            for c in captions:
                tokenized = _python_tokenize(c['caption'].replace('\n', ' '))
                final_tokenized_captions_for_audio[k].append(tokenized)
        return final_tokenized_captions_for_audio
