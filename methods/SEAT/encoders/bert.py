""" Convenience functions for handling BERT """
import os
import numpy as np

from transformers import BertTokenizer, BertModel
from transformers import BertTokenizerFast

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load BERT model (bbc) and corresponding tokenizer from local files """

    model = BertModel.from_pretrained(models_dir + '/bert/base-cased/')
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(models_dir + '/bert/base-cased/')
    # additional 'Fast' BERT tokenizer for subword tokenization ID mapping
    subword_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    return model, tokenizer, subword_tokenizer

def encode(model, tokenizer, subword_tokenizer, sents, stimuli, encoding, multiple_words):
    """ Function to encode sentences with BERT """

    encs = {}
    for sent in sents:
        # [CLS] and [SEP] tokens are added automatically
        token_ids = tokenizer(sent[:-1], return_tensors='pt')
        # map tokens to input words
        subword_ids = subword_tokenizer(sent[:-1], add_special_tokens=False).word_ids()
        vecs = model(**token_ids)
        encoding_level = encoding[:4]
        if encoding_level == 'word': # here: subword tokenization
            if multiple_words: # case: multiple words
                # determine idx of stimuli in input sentence
                stimulus = [stimulus for stimulus in stimuli if stimulus in sent][0]
                # account for [CLS] token
                idx_start = sent[:-1].split().index(stimulus.split()[0]) + 1
                # account for [CLS] token; range function excludes end idx
                idx_end = idx_start + len(stimulus.split())
                # obtain vecs of all relevant tokens
                token_vecs = []
                for idxs in range(idx_start, idx_end):
                    token_vecs.append(vecs['last_hidden_state'][0][idxs].detach().numpy())
                # extract rep of token of interest as average over all tokens
                encs[sent] = np.mean(np.asarray(token_vecs), axis=0)
            else:
                # determine idx of stimulus in input sentence
                idx = None
                tokens = sent[:-1].split()
                for i, token in enumerate(tokens):
                    if token in stimuli:
                        idx = i
                if '-' in tokens[idx]: # case: special example of subword tokenization
                    idx_stimuli = [i for i, element in enumerate(subword_ids) if element == idx]
                    # account for [CLS] token
                    idx_start = idx_stimuli[0] + 1
                    idxs_first_part = len(idx_stimuli)
                    idxs_second_part = len([i for i, element in enumerate(subword_ids) if element == (idx_start+1)])
                    # account for [CLS] token; range function excludes end idx
                    idx_end = idx_start + idxs_first_part + idxs_second_part + 1
                    if encoding == 'word-average':
                        # obtain vecs of all relevant tokens
                        token_vecs = []
                        for idxs in range(idx_start, idx_end):
                            token_vecs.append(vecs['last_hidden_state'][0][idxs].detach().numpy())
                        # extract rep of token of interest as average over all tokens
                        encs[sent] = np.mean(np.asarray(token_vecs), axis=0)
                    elif encoding == 'word-start':
                        encs[sent] = vecs['last_hidden_state'][0][idx_start].detach().numpy()
                    elif encoding == 'word-end':
                        idx_new = idx_start + idxs_first_part + idxs_second_part
                        encs[sent] = vecs['last_hidden_state'][0][idx_new].detach().numpy()
                else:
                    if subword_ids.count(idx) == 1: # case: no subword tokenization
                        # account for [CLS] token
                        idx_new = idx + 1
                        # extract rep of token of interest
                        encs[sent] = vecs['last_hidden_state'][0][idx_new].detach().numpy()
                    elif subword_ids.count(idx) > 1: # case: subword tokenization
                        if encoding == 'word-average':
                            # obtain vecs of all relevant subwords
                            subword_vecs = []
                            idx_list = [i for i in range(len(subword_ids)) if subword_ids[i] == idx]
                            for idxs in idx_list:
                                # account for [CLS] token
                                idx_new = idxs + 1
                                subword_vecs.append(vecs['last_hidden_state'][0][idx_new].detach().numpy())
                            # extract rep of token of interest as average over all subwords
                            encs[sent] = np.mean(np.asarray(subword_vecs), axis=0)
                        elif encoding == 'word-start':
                            # account for CLS token
                            idx_new = subword_ids.index(idx) + 1
                            # extract rep of token of interest as first subword
                            encs[sent] = vecs['last_hidden_state'][0][idx_new].detach().numpy()
                        elif encoding == 'word-end':
                            # account for [CLS] token
                            idx_new = len(subword_ids) - subword_ids[::-1].index(idx)
                            # extract rep of token of interest as last subword
                            encs[sent] = vecs['last_hidden_state'][0][idx_new].detach().numpy()
        elif encoding_level == 'sent':
            # extract rep of sent as [CLS] token
            encs[sent] = vecs['last_hidden_state'][0][0].detach().numpy()

    return encs
