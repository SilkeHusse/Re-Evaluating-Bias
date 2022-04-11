""" Convenience functions for handling GPT2 """
import os
import numpy as np

from transformers import GPT2Model, GPT2Tokenizer
from transformers import GPT2TokenizerFast

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load GPT2 model and corresponding tokenizer from local files """

    model = GPT2Model.from_pretrained(models_dir + '/gpt2/')
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(models_dir + '/gpt2/')
    # additional 'Fast' GPT2 tokenizer for subword tokenization ID mapping
    subword_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return model, tokenizer, subword_tokenizer

def encode(model, tokenizer, subword_tokenizer, sents, stimuli, encoding, multiple_words):
    """ Function to encode sentences with GPT2 """

    encs = {}
    for sent in sents:
        token_ids = tokenizer(sent[:-1], return_tensors='pt')
        # map tokens to input words
        subword_ids = subword_tokenizer(sent[:-1], add_special_tokens=False).word_ids()
        vecs = model(**token_ids)
        encoding_level = encoding[:4]
        if encoding_level == 'word': # here: subword tokenization
            if multiple_words: # case: multiple words
                # determine idx of stimuli in input sentence
                stimulus = [stimulus for stimulus in stimuli if stimulus in sent][0]
                idx_start = sent[:-1].split().index(stimulus.split()[0])
                # range function excludes end idx
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
                    idx_start = idx_stimuli[0]
                    idxs_first_part = len(idx_stimuli)
                    idxs_second_part = len([i for i, element in enumerate(subword_ids) if element == (idx_start + 2)])
                    # range function excludes end idx
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
                        # extract rep of token of interest
                        encs[sent] = vecs['last_hidden_state'][0][idx].detach().numpy()
                    elif subword_ids.count(idx) > 1: # case: subword tokenization
                        if encoding == 'word-average':
                            # obtain vecs of all relevant subwords
                            subword_vecs = []
                            idx_list = [i for i in range(len(subword_ids)) if subword_ids[i] == idx]
                            for idxs in idx_list:
                                subword_vecs.append(vecs['last_hidden_state'][0][idxs].detach().numpy())
                            # extract rep of token of interest as average over all subwords
                            encs[sent] = np.mean(np.asarray(subword_vecs), axis=0)
                        elif encoding == 'word-start':
                            idx_subword = subword_ids.index(idx)
                            # extract rep of token of interest as first subword
                            encs[sent] = vecs['last_hidden_state'][0][idx_subword].detach().numpy()
                        elif encoding == 'word-end':
                            idx_subword = len(subword_ids) - 1 - subword_ids[::-1].index(idx)
                            # extract rep of token of interest as last subword
                            encs[sent] = vecs['last_hidden_state'][0][idx_subword].detach().numpy()

        elif encoding_level == 'sent':
            idx_word = len(vecs['last_hidden_state'][0]) - 1
            # extract rep of sent as last token
            encs[sent] = vecs['last_hidden_state'][0][idx_word].detach().numpy()
    return encs
