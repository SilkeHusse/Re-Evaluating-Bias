""" Implementation of log prob bias score calculations """
import pandas as pd
import numpy as np
import torch

torch.manual_seed(1111)
np.random.seed(1111)

def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)

def elmo_logits(sent, targ_wds, model, apply_softmax=True):
    """ Function to obtain logits for ELMo """
    pass # TODO
    #result = {w: float() for w in targ_wds}
    #for targ_wd in targ_wds:
        #sent_replaced = sent.replace('TTT', targ_wd)
        #logits = model.embed_sentence(sent_replaced)
    return None#result

def bert_logits(template, sent, targ_wds, model, tok, subword_tok, apply_softmax=True):
    """ Function to obtain (prior) logits for BERT """

    token_ids = tok(sent[:-1], return_tensors='pt') # [CLS] and [SEP] tokens are added automatically
    # map tokens to input words
    subword_ids = subword_tok(sent[:-1], add_special_tokens=False).word_ids()
    outputs = model(**token_ids, labels=token_ids["input_ids"])
    logits = outputs.logits[0,:,:].detach().numpy()
    if apply_softmax:
        logits = softmax(logits)

    # determine idx of [MASK] in input sentence
    # case: AAA occurs before TTT and thus take second / last occurrence of [MASK] token
    if sent[:-1].split().count('[MASK]') > 1 and template[:-1].split().index('AAA') < template[:-1].split().index('TTT'):
        idx_mask = None
        tokens = sent[:-1].split()
        for i, token in enumerate(tokens):
            if token == '[MASK]':
                idx_mask = i
    # case: TTT occurs before AAA and thus take first occurrence of [MASK] token
    else:
        idx_mask = sent[:-1].split().index('[MASK]')
    # case: subword tokenization before or after [MASK] token
    if len(sent[:-1].split()) != len(subword_ids):
        idx_mask = [i for i in range(len(subword_ids)) if subword_ids[i] == idx_mask][0] + 1 # account for CLS token
    # case: no subword tokenization in input sentence
    else:
        idx_mask += 1 # account for CLS token

    result = {w: float() for w in targ_wds}
    for targ_wd in targ_wds:
        # case: target word consists of subwords or multiple words
        if len(subword_tok(targ_wd, add_special_tokens=False).word_ids()) > 1:
           subword_token_ids = subword_tok(targ_wd, add_special_tokens=False)['input_ids']
           subwords = [k for k, v in tok.vocab.items() if v in subword_token_ids]
           subwords_logits = [logits[idx_mask, tok.vocab[subwd]] for subwd in subwords]
           result[targ_wd] = np.nanprod(subwords_logits)  # take prod of all probs
           # here: word encoding level
           #if encoding == 'word-start':
           #    subword = [k for k, v in tok.vocab.items() if v == subword_token_ids[0]][0]
           #    result[targ_wd] = logits[idx_mask, tok.vocab[subword]]
           #elif encoding == 'word-end':
           #    subword = [k for k, v in tok.vocab.items() if v == subword_token_ids[-1]][0]
           #    result[targ_wd] = logits[idx_mask, tok.vocab[subword]]
           #elif encoding == 'word-average':
           #     subwords = [k for k, v in tok.vocab.items() if v in subword_token_ids]
           #     subwords_logits = [logits[idx_mask, tok.vocab[subwd]] for subwd in subwords]
           #     result[targ_wd] = np.prod(subwords_logits) # take prod of all probs
           #else:
           #    raise ValueError("Encoding level %s not found!" % encoding)
        else:
            result[targ_wd] = logits[idx_mask, tok.vocab[targ_wd]]

    return result

def gpt2_logits(sent, targ_wds, model, tok, apply_softmax=True):
    """ Function to obtain logits for GPT2 """

    result = {w: float() for w in targ_wds}
    for targ_wd in targ_wds:
        sent_replaced = sent.replace('TTT', targ_wd)
        token_ids = tok(sent_replaced[:-1], return_tensors='pt')
        outputs = model(**token_ids, labels=token_ids["input_ids"])
        logits = outputs.logits[0, :, :].detach().numpy()
        if apply_softmax:
            logits = softmax(logits)

        probs = []
        for idx in range(len(token_ids['input_ids'][0])):
            token_id = token_ids['input_ids'][0][idx]
            probs.append(logits[idx, token_id])
        result[targ_wd] = np.nanprod(probs)

    return result

def gpt2_prior_logits(sent, targ_wds, model, tok, subword_tok, apply_softmax=True):
    """ Function to obtain prior logits for GPT2 """

    # specify sentence to original length
    len_sent = len(set(subword_tok(sent[:-1], add_special_tokens=False).word_ids()))
    result = {w: float() for w in targ_wds}
    for targ_wd in targ_wds:
        len_sent += len(tok(targ_wd, return_tensors='pt')['input_ids'][0])

        # execute top-p nucleus sampling
        token_ids = tok(targ_wd, return_tensors='pt')
        sample_outputs = model.generate(token_ids['input_ids'],
            do_sample=True,
            max_length=len_sent,
            top_p=0.95,
            top_k=0,
            num_return_sequences = 10
        )

        probs_total = []
        for i, sample_output in enumerate(sample_outputs):
            # compute prob of full sentence
            outputs = model(sample_output)
            logits = outputs.logits.detach().numpy()
            if apply_softmax:
                logits = softmax(logits)

            probs = []
            for idx in range(len(sample_output)):
                token_id = sample_output[idx]
                probs.append(logits[idx, token_id])
            probs_total.append(np.nanprod(probs))

        # combine all probs via mean over all sample sentences
        result[targ_wd] = np.nanmean(probs_total)

    return result

def bias_score(template_sent, targ1_wds, targ2_wds, attr_wd, model_name, model, tok, subword_tok):
    """ Function to compute the log prob bias score
        args:
        - template_sent (str): template sent containing 'TTT' and 'AAA'
        - targ1_wds (list(str)): list containing stimuli from target1 (X)
        - targ2_wds (list(str)): list containing stimuli from target2 (Y)
        - attr_wd (str): attribute word of interest
        - model_name (str): name of specified LM
        - model, tok, subword_tok: specified LM including tokenizers
    """

    # p_tgt : prob of filling [MASK] token with target words given sent with attribute word
    targ_wds = [wd.lower() for wd in targ1_wds+targ2_wds]
    if model_name == 'elmo':
        logits_tgt = elmo_logits(template_sent.replace('AAA', attr_wd), targ_wds, model)
    elif model_name == 'bert':
        logits_tgt = bert_logits(template_sent, template_sent.replace('AAA', attr_wd).replace('TTT', '[MASK]'),
                                          targ_wds, model, tok, subword_tok)
    elif model_name == 'gpt2':
        logits_tgt = gpt2_logits(template_sent.replace('AAA', attr_wd), targ_wds, model, tok)

    bias = np.log(sum(logits_tgt[wd.lower()] for wd in targ1_wds)) - \
                        np.log(sum(logits_tgt[wd.lower()] for wd in targ2_wds))

    # p_prior : prob of filling [MASK] token with target words given sent with masked attribute word
    if model_name == 'elmo':
        logits_prior = elmo_prior_logits() #TODO
    elif model_name == 'bert':
        logits_prior = bert_logits(template_sent, template_sent.replace('AAA', '[MASK]').replace('TTT', '[MASK]'),
                                                targ_wds, model, tok, subword_tok)
    elif model_name == 'gpt2':
        logits_prior = gpt2_prior_logits(template_sent.replace('AAA', attr_wd), targ_wds, model, tok, subword_tok)

    bias_prior_correction = np.log(sum(logits_prior[wd.lower()] for wd in targ1_wds)) - \
                                         np.log(sum(logits_prior[wd.lower()] for wd in targ2_wds))

    return {"stimulus": attr_wd,
            "bias": bias,
            "prior_correction": bias_prior_correction,
            "bias_prior_corrected": bias - bias_prior_correction}

def exact_mc_perm_test(xs, ys, nmc=100000):
    """ Function to compute p-value """
    n, k = len(xs), 0
    s = np.abs(np.mean(xs) - np.mean(ys)) # two-sided p-value
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs) # permutation test
        k += s < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

def logprob_cal(model_name, model, tokenizer, subword_tokenizer, final_template):
    """ Function to run log prob bias score calculations
        args:
            - model_name (str): name of specified LM
            - model, tokenizer, subword_tokenizer: specified LM including tokenizers
            - final_template (Dict[int: Dict]): dictionary mapping template sentences with respective stimuli
    """

    df_attr1_lst, df_attr2_lst = [], []
    for key, value in final_template.items():
        results = [bias_score(final_template[key]['template'],
                              final_template[key]['stimuli_targ1'],
                              final_template[key]['stimuli_targ2'],
                              w, model_name, model, tokenizer, subword_tokenizer) for w in final_template[key]['stimuli_attr1']]
        df_attr1_lst.append(pd.DataFrame(results))
    for key, value in final_template.items():
        results = [bias_score(final_template[key]['template'],
                              final_template[key]['stimuli_targ1'],
                              final_template[key]['stimuli_targ2'],
                              w, model_name, model, tokenizer, subword_tokenizer) for w in final_template[key]['stimuli_attr2']]
        df_attr2_lst.append(pd.DataFrame(results))
    df1, df2 = pd.concat(df_attr1_lst), pd.concat(df_attr2_lst)

    k = 'bias_prior_corrected'
    std_AB = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
    esize = (df1[k].mean() - df2[k].mean()) / std_AB
    pvalue = exact_mc_perm_test(df1[k], df2[k])

    return esize, pvalue
