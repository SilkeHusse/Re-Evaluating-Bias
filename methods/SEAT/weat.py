""" Implementation of WEAT """
import math
import itertools as it
import numpy as np
import scipy.special
import scipy.stats

np.random.seed(1111)
# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.

def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))
def construct_cossim_lookup(XY, AB):
    """ Function to compute cosine similarities"""
    cossims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
    return cossims
def s_wAB(A, B, cossims):
    """ Function for
    s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)
    """
    return cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)
def s_XAB(X, s_wAB_memo):
    """ Function for single term of test statistic
    sum_{x in X} s(x, A, B)
    """
    return s_wAB_memo[X].sum()
def s_XYAB(X, Y, s_wAB_memo):
    """ Function for test statistic
    s(X, Y, A, B) = sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
    """
    return s_XAB(X, s_wAB_memo) - s_XAB(Y, s_wAB_memo)
def mean_s_wAB(X, A, B, cossims):
    return np.mean(s_wAB(A, B, cossims[X]))
def stdev_s_wAB(X, A, B, cossims):
    return np.std(s_wAB(A, B, cossims[X]), ddof=1)

def convert_keys_to_ints(X, Y):
    return (dict((i, v) for (i, (k, v)) in enumerate(X.items())),
            dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),)

def p_val_permutation_test(X, Y, A, B, n_samples, cossims, parametric):
    """ Function to compute the p-value for the permutation test
    Pr[ s(Xi, Yi, A, B) ≥ s(X, Y, A, B) ]
    for Xi, Yi : partition of X union Y
    """
    X = np.array(list(X), dtype=np.int)
    Y = np.array(list(Y), dtype=np.int)
    A = np.array(list(A), dtype=np.int)
    B = np.array(list(B), dtype=np.int)

    size = len(X)
    s_wAB_memo = s_wAB(A, B, cossims=cossims)
    XY = np.concatenate((X, Y))

    if parametric: # case: assume normal distribution
        s = s_XYAB(X, Y, s_wAB_memo)
        samples = []
        for _ in range(n_samples): # permutation test
            np.random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            si = s_XYAB(Xi, Yi, s_wAB_memo)
            samples.append(si)
        # unbiased mean and standard deviation
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        return p_val

    else: # case: non-parametric implementation
        s = s_XAB(X, s_wAB_memo)
        total_true, total_equal, total = 0, 0, 0
        num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
        if num_partitions > n_samples:
            # draw 99,999 samples and bias by 1 positive observation
            total_true += 1
            total += 1
            for _ in range(n_samples - 1):
                np.random.shuffle(XY)
                Xi = XY[:size]
                #Yi = XY[size:]
                si = s_XAB(Xi, s_wAB_memo)
                if si > s: # case: strict inequality
                    total_true += 1
                elif si == s:  # case: conservative non-strict inequality
                    total_true += 1
                    total_equal += 1
                total += 1
        else:  # case: use exact permutation test (number of partitions)
            for Xi in it.combinations(XY, len(X)):
                Xi = np.array(Xi, dtype=np.int)
                #Yi = np.asarray([i for i in XY if i not in Xi])
                si = s_XAB(Xi, s_wAB_memo)
                if si > s: # case: strict inequality
                    total_true += 1
                elif si == s:  # case: conservative non-strict inequality
                    total_true += 1
                    total_equal += 1
                total += 1
        #print('Equalities contributed {}/{} to p-value'.format(total_equal, total))
        return total_true / total

def effect_size(X, Y, A, B, cossims):
    """ Function to compute the effect size
    [ mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B) ] /
        [ stddev_{w in X union Y} s(w, A, B) ]
    """
    X = list(X)
    Y = list(Y)
    A = list(A)
    B = list(B)

    numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(Y, A, B, cossims=cossims)
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    return numerator / denominator

def run_test(encs, parametric, n_samples=100000):
    """ Function to run a WEAT test
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the concept and the encodings
        - parametric (bool): execute (non)-parametric version of test
        - n_samples (int): number of samples to draw to estimate p-value
    """
    X, Y = encs["targ1"]['encs'], encs["targ2"]['encs']
    A, B = encs["attr1"]['encs'], encs["attr2"]['encs']
    # convert keys to ints for easier array lookups
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)
    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    p_val = p_val_permutation_test(X, Y, A, B, n_samples=n_samples, cossims=cossims, parametric=parametric)
    esize = effect_size(X, Y, A, B, cossims=cossims)
    return esize, p_val
