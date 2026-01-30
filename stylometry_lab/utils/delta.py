import numpy as np
from collections import Counter

def burrows_delta(tokensA, tokensB, tokensU, top_n=200):
    freq = Counter(tokensA + tokensB + tokensU)
    vocab = [w for w, _ in freq.most_common(top_n)]

    def rel_freq(tokens):
        c = Counter(tokens)
        tot = max(1, len(tokens))
        return np.array([c[w] / tot for w in vocab])

    vA = rel_freq(tokensA)
    vB = rel_freq(tokensB)
    vU = rel_freq(tokensU)

    mean = np.mean([vA, vB], axis=0)
    sd = np.std(np.vstack([vA, vB]), axis=0)
    sd[sd == 0] = 1e-8

    zA = (vA - mean) / sd
    zB = (vB - mean) / sd
    zU = (vU - mean) / sd

    deltaA = float(np.mean(np.abs(zU - zA)))
    deltaB = float(np.mean(np.abs(zU - zB)))

    return deltaA, deltaB, vocab
