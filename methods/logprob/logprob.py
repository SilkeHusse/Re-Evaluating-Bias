import pandas as pd
import numpy as np

np.random.seed(1111)

def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)

def get_mask_fill_logits(template, sent, targ_wds, model, tok, subword_tok, encoding, apply_softmax=True):

    token_ids = tok(sent[:-1], return_tensors='pt') # [CLS] and [SEP] tokens are added automatically
    # map tokens to input words
    subword_ids = subword_tok(sent[:-1], add_special_tokens=False).word_ids()
    outputs = model(**token_ids, labels=token_ids["input_ids"])
    logits = outputs.logits[0,:,:].detach().numpy()
    if apply_softmax:
        logits = softmax(logits)

    if sent[:-1].split().count('[MASK]') > 1 and template[:-1].split().index('AAA') < template[:-1].split().index('TTT'):
        # determine idx of [MASK] in input sentence
        idx_mask = None # find second / last occurence dh AAA before TTT
        tokens = sent[:-1].split()
        for i, token in enumerate(tokens):
            if token == '[MASK]':
                idx_mask = i
    else:
        # determine idx of [MASK] in input sentence
        idx_mask = sent[:-1].split().index('[MASK]') # finds first occurence !! dh TTT before AAA
    if len(sent[:-1].split()) != len(subword_ids): # case: subword tokenization before or after [MASK] token
        # account for CLS token
        idx_mask = [i for i in range(len(subword_ids)) if subword_ids[i] == idx_mask][0] + 1
    else: # case: no subword tokenization in input sentence
        idx_mask += 1  # account for CLS token

    result = {w: float() for w in targ_wds}
    for targ_wd in targ_wds:
        #if len(subword_tok(targ_wd, add_special_tokens=False).word_ids()) > 1: # case: targ word has subword tokenization
        #    subword_token_ids = subword_tok(targ_wd, add_special_tokens=False)['input_ids']
        #    if encoding == 'word-start':
        #        subword = [k for k, v in tok.vocab.items() if v == subword_token_ids[0]][0]
        #        result[targ_wd] = logits[idx_mask, tok.vocab[subword]]
        #    elif encoding == 'word-end':
        #        subword = [k for k, v in tok.vocab.items() if v == subword_token_ids[-1]][0]
        #        result[targ_wd] = logits[idx_mask, tok.vocab[subword]]
        #    elif encoding == 'word-average':
        #        subwords = [k for k, v in tok.vocab.items() if v in subword_token_ids]
        #        subwords_logits = [logits[idx_mask, tok.vocab[subwd]] for subwd in subwords]
        #        result[targ_wd] = np.mean(subwords_logits)
        #    else:
        #        raise ValueError("Encoding level %s not found!" % encoding)
        #else:
        result[targ_wd] = logits[idx_mask, tok.vocab[targ_wd]]

    return result

def bias_score(template_sent, targ1_wds, targ2_wds, attr_wd, model, tok, subword_tok, encoding, original):
    """
    Input a sentence of the form "TTT is AAA"
    AAA is a placeholder for the target word
    TTT is a placeholder for the gendered words (the subject)
    We will predict the bias when filling in the gendered words and
    filling in the target word.
    """
    # probability of filling [MASK] with "he" vs. "she" when target is "programmer"
    targ_wds = targ1_wds + targ2_wds

    if original:
        subject_fill_logits = get_mask_fill_logits(
            template_sent, template_sent.replace('AAA', attr_wd).replace('TTT', '[MASK]'), targ_wds,
            model, tok, subword_tok, encoding)
        subject_fill_bias = np.log(sum(subject_fill_logits[wd] for wd in targ1_wds)) - \
                            np.log(sum(subject_fill_logits[wd] for wd in targ2_wds))
        # male words are simply more likely than female words
        # correct for this by masking the target word and measuring the prior probabilities
        subject_fill_prior_logits = get_mask_fill_logits(
            template_sent, template_sent.replace('AAA', '[MASK]').replace('TTT', '[MASK]'), targ_wds,
            model, tok, subword_tok, encoding)
        subject_fill_bias_prior_correction = np.log(sum(subject_fill_prior_logits[wd] for wd in targ1_wds)) - \
                                         np.log(sum(subject_fill_prior_logits[wd] for wd in targ2_wds))
    else:
        pass
        # TODO: modified log prob method

    return {
        "stimulus": attr_wd,
        "bias": subject_fill_bias,
        "prior_correction": subject_fill_bias_prior_correction,
        "bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
    }

def exact_mc_perm_test(xs, ys, nmc=100000):
    n, k = len(xs), 0
    s = np.abs(np.mean(xs) - np.mean(ys)) # two-sided test of permutation test
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += s < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

def logprob_cal(model, tokenizer, subword_tokenizer, encoding, final_template, original):
    """ Function to run logprob bias score calculations
        args:
            - model, tokenizer, subword_tokenizer (str): specifies LM including tokenizers to load
            - encodings ([str]): list containing encoding levels
            - final_template (Dict[int: Dict]): dictionary mapping template sentences with respective stimuli
            - original (bool): indicates if original (True) or modified (False) logprob method will be used
    """

    # TODO: encoding level parameter

    df_attr1_lst, df_attr2_lst = [], []
    for key, value in final_template.items():
        results = [bias_score(final_template[key]['template'],
                              final_template[key]['stimuli_targ1'],
                              final_template[key]['stimuli_targ2'], w, model, tokenizer, subword_tokenizer, encoding, original)
                   for w in final_template[key]['stimuli_attr1']]
        df_attr1_lst.append(pd.DataFrame(results))
    for key, value in final_template.items():
        results = [bias_score(final_template[key]['template'],
                              final_template[key]['stimuli_targ1'],
                              final_template[key]['stimuli_targ2'], w, model, tokenizer, subword_tokenizer, encoding, original)
                   for w in final_template[key]['stimuli_attr2']]
        df_attr2_lst.append(pd.DataFrame(results))
    df1 = pd.concat(df_attr1_lst)
    df2 = pd.concat(df_attr2_lst)

    k = 'bias_prior_corrected'
    std_AB = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
    esize = (df1[k].mean() - df2[k].mean()) / std_AB
    pvalue = exact_mc_perm_test(df1[k], df2[k])

    return esize, pvalue
