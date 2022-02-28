""" Convenience functions for handling GPT2 """
import os
import numpy as np

from transformers import GPT2Model, GPT2Tokenizer
from transformers import GPT2TokenizerFast

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load BERT model (bbc) and corresponding tokenizer from local files  """

    model = GPT2Model.from_pretrained(models_dir + '/gpt2/')
    tokenizer = GPT2Tokenizer.from_pretrained(models_dir + '/gpt2/')
    # additional 'Fast' GPT2 tokenizer for subword tokenization ID mapping
    subword_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return model, tokenizer, subword_tokenizer

def encode(model, tokenizer, subword_tokenizer, sents, stimulis, encoding):
    """ Function to encode sentences with GPT2 """

    encs = {}
    for sent in sents:
        token_ids = tokenizer(sent[:-1], return_tensors='pt')
        subword_ids = subword_tokenizer(sent[:-1], add_special_tokens=False).word_ids() # map tokens to input words
        vecs = model(**token_ids)
        encoding_level = encoding[:4]
        if encoding_level == 'word': # here: subword tokenization
            # determine idx of words in input sentence
            idx = None
            tokens = sent[:-1].split()
            for i, token in enumerate(tokens):
                if token.lower() in stimulis:
                    idx = i

            if subword_ids.count(idx) == 1: # case: no subword tokenization
                encs[sent] = vecs['last_hidden_state'][0][idx].detach().numpy() # extract rep of token of interest
            elif subword_ids.count(idx) > 1: # case: subword tokenization
                if encoding == 'word-average':
                    subword_vecs = []  # obtain vecs of all relevant subwords
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
                    idx_subword = len(subword_ids) -1 - subword_ids[::-1].index(idx)
                    # extract rep of token of interest as last subword
                    encs[sent] = vecs['last_hidden_state'][0][idx_subword].detach().numpy()

        elif encoding_level == 'sent':
            idx_word = len(vecs['last_hidden_state'][0]) - 1
            encs[sent] = vecs['last_hidden_state'][0][idx_word].detach().numpy() # extract rep of sent as last token
    return encs
