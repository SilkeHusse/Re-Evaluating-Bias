""" Implementation of CEAT calculations """
import numpy as np
import scipy.stats
import random

random.seed(1111)

from sklearn.metrics.pairwise import cosine_similarity

def associate(w,A,B):
    """ Function for
    s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)
    """
    return cosine_similarity(w.reshape(1,-1),A).mean() - cosine_similarity(w.reshape(1,-1),B).mean()

def effect_size(X,Y,A,B):
    """ Function to compute the effect size
    [ mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B) ] /
        [ stddev_{w in X union Y} s(w, A, B) ]
    """
    delta_mean =  np.mean([associate(X[i,:],A,B) for i in range(X.shape[0])]) - np.mean([associate(Y[i,:],A,B) for i in range(Y.shape[0])])

    XY = np.concatenate((X,Y),axis=0)
    s_union = [associate(XY[i,:],A,B) for i in range(XY.shape[0])]
    std_dev = np.std(s_union,ddof=1)
    var = std_dev**2

    return delta_mean/std_dev, var

def ceat_meta(encs, encoding, N=10000):
    """ Function to run CEAT calculations
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
        to dictionaries containing the concept and the encodings, the encodings are further saved
        in dictionaries with the relevant stimuli as key and dictionaries for each encoding level as value
        example:
        encs['targ1'] = {'concept': 'Flower_name', 'encs': encs_targ1}
        encs_targ1 = {'aster': {
                        'sent': [sents],
                        'word-average': [sents],
                        'word-start': [sents],
                        'word-end': [sents]     },
                      ...  }
        - encoding (str): specifies encoding level
        - N (int): number of effect size samples
    """

    weat_dict_targ1 = {wd: encs['targ1']['encs'][wd][encoding] for wd in list(encs['targ1']['encs'].keys())}
    weat_dict_targ2 = {wd: encs['targ2']['encs'][wd][encoding] for wd in list(encs['targ2']['encs'].keys())}
    weat_dict_attr1 = {wd: encs['attr1']['encs'][wd][encoding] for wd in list(encs['attr1']['encs'].keys())}
    weat_dict_attr2 = {wd: encs['attr2']['encs'][wd][encoding] for wd in list(encs['attr2']['encs'].keys())}

    e_lst = [] # list containing N effect sizes
    v_lst = [] # list containing corresponding N variances

    weat_dicts = [weat_dict_targ1, weat_dict_targ2, weat_dict_attr1, weat_dict_attr2]
    sents_idx_lsts = []
    for weat_dict in weat_dicts:
        sents_dict = {}
        for wd in list(weat_dict.keys()):
            if len(weat_dict[wd]) < N : # case: sample with replacement
                idx_sents = random.choices(range(len(weat_dict[wd])), k=N)
            else: # case: sample without replacement
                idx_sents = random.sample(range(len(weat_dict[wd])), N)
            sents_dict[wd] = idx_sents
        sents_idx_lsts.append(sents_dict)

    for i in range(N):
        X = np.array([weat_dict_targ1[wd][sents_idx_lsts[0][wd][i]] for wd in list(weat_dict_targ1.keys())])
        Y = np.array([weat_dict_targ2[wd][sents_idx_lsts[1][wd][i]] for wd in list(weat_dict_targ2.keys())])
        A = np.array([weat_dict_attr1[wd][sents_idx_lsts[2][wd][i]] for wd in list(weat_dict_attr1.keys())])
        B = np.array([weat_dict_attr2[wd][sents_idx_lsts[3][wd][i]] for wd in list(weat_dict_attr2.keys())])
        e,v = effect_size(X,Y,A,B)
        e_lst.append(e)
        v_lst.append(v)

    # random-effects model from meta-analysis literature
    e_array = np.array(e_lst)
    w_array = 1 / np.array(v_lst)

    # total variance Q
    q1 = np.sum(w_array*(e_array**2))
    q2 = ((np.sum(e_array*w_array))**2)/np.sum(w_array)
    q = q1 - q2
    df = N - 1 # degrees of freedom

    # variance decomposition:
    # if only source of variance is within-study var then Q = df
    # thus compute between-studies var tao_square (excess variance)
    if q > df:
        # c : scaling factor such that tao_square is in same metric as within-study var
        c = np.sum(w_array) - np.sum(w_array**2)/np.sum(w_array)
        tao_square = (q-df)/c
    else:
        tao_square = 0

    # weighting of each study by the inverse of its variance
    v_star_array = np.array(v_lst) + tao_square # var includes all variance components
    w_star_array = 1/v_star_array

    # combined effect size (weighted mean)
    ces = np.sum(w_star_array*e_array)/np.sum(w_star_array)
    v = 1/np.sum(w_star_array)
    z = ces/np.sqrt(v)
    # 2-tailed p-value, standard normal cdf (by CLS)
    #p_value = scipy.stats.norm.sf(z, loc = 0, scale = 1)
    p_value = 2 * scipy.stats.norm.sf(abs(z), loc = 0, scale = 1)

    return ces, p_value