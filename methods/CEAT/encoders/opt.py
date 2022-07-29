""" Convenience functions for handling OPT """
import os
import numpy as np

from transformers import OPTModel, GPT2Tokenizer
from transformers import GPT2TokenizerFast

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load OPT model and corresponding tokenizer from local files """

    model = OPTModel.from_pretrained(models_dir + '/opt/')
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(models_dir + '/opt/')
    # additional 'Fast' GPT2 tokenizer for subword tokenization ID mapping
    subword_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return model, tokenizer, subword_tokenizer

def encode(model, tokenizer, subword_tokenizer, sents, stimuli, encodings):
    """ Function to encode sentences with OPT """

    encs = {i:{'sent': [], 'word-average': [], 'word-start': [], 'word-end': []} for i in stimuli}
    for wd in stimuli:
        for sent in sents[wd]:
            token_ids = tokenizer(sent[:-1], return_tensors='pt') # [BOS] token is added automatically
            subword_ids = subword_tokenizer(sent[:-1], add_special_tokens=False).word_ids() # map tokens to input words
            vecs = model(**token_ids)

            for encoding in encodings:
                encoding_level = encoding[:4]

                if encoding_level == 'word': # here: subword tokenization

                    if len(wd.split()) > 1: # case: multiple words
                        # determine idx of stimuli in input sentence
                        stimulus = [stimulus for stimulus in stimuli if stimulus in sent][0]
                        idx_start = sent[:-1].split().index(stimulus.split()[0]) + 1 # account for [BOS] token
                        idx_end = idx_start + len(stimulus.split()) # account for [BOS] token; range function excludes end idx

                        # extract rep of token of interest as average over all relevant tokens
                        vecs_token = []
                        for idx_token in range(idx_start, idx_end):
                            vecs_token.append(vecs['last_hidden_state'][0][idx_token].detach().numpy())
                        encs[wd][encoding].append(np.mean(np.asarray(vecs_token), axis=0))

                    else:
                        # determine idx of stimulus in input sentence
                        idx = None
                        tokens = sent[:-1].split()
                        for i, token in enumerate(tokens):
                            if token in stimuli:
                                idx = i

                        if '-' in tokens[idx]: # case: special example of subword tokenization
                            idx_stimuli = [i for i, element in enumerate(subword_ids) if element == idx]
                            idx_start = idx_stimuli[0] + 1 # account for [BOS] token
                            len_first_part = len(idx_stimuli)
                            len_second_part = len([i for i, element in enumerate(subword_ids) if element == (idx + 2)])
                            # account for [BOS] token; range function excludes end idx
                            idx_end = idx_start + len_first_part + len_second_part + 1

                            if encoding == 'word-average':
                                # extract rep of token of interest as average over all relevant tokens
                                vecs_token = []
                                for idx_token in range(idx_start, idx_end):
                                    vecs_token.append(vecs['last_hidden_state'][0][idx_token].detach().numpy())
                                encs[wd][encoding].append(np.mean(np.asarray(vecs_token), axis=0))
                            elif encoding == 'word-start':
                                encs[wd][encoding].append(vecs['last_hidden_state'][0][idx_start].detach().numpy())
                            elif encoding == 'word-end':
                                idx_end = idx_end - 1
                                encs[wd][encoding].append(vecs['last_hidden_state'][0][idx_end].detach().numpy())

                        else:

                            if subword_ids.count(idx) == 1: # case: no subword tokenization
                                idx_new = idx + 1 # account for [BOS] token
                                # extract rep of token of interest
                                encs[wd][encoding].append(vecs['last_hidden_state'][0][idx_new].detach().numpy())

                            elif subword_ids.count(idx) > 1: # case: subword tokenization

                                if encoding == 'word-average':
                                    # obtain vecs of all relevant subwords
                                    vecs_subword = []
                                    idx_list = [i for i in range(len(subword_ids)) if subword_ids[i] == idx]
                                    for idx in idx_list:
                                        idx_new = idx + 1 # account for [BOS] token
                                        vecs_subword.append(vecs['last_hidden_state'][0][idx_new].detach().numpy())
                                    # extract rep of token of interest as average over all subwords
                                    encs[wd][encoding].append(np.mean(np.asarray(vecs_subword), axis=0))
                                elif encoding == 'word-start':
                                    idx_new = subword_ids.index(idx) + 1 # account for [BOS] token
                                    # extract rep of token of interest as first subword
                                    encs[wd][encoding].append(vecs['last_hidden_state'][0][idx_new].detach().numpy())
                                elif encoding == 'word-end':
                                    idx_new = len(subword_ids) - subword_ids[::-1].index(idx) # account for [BOS] token
                                    # extract rep of token of interest as last subword
                                    encs[wd][encoding].append(vecs['last_hidden_state'][0][idx_new].detach().numpy())

                elif encoding_level == 'sent':
                    idx_new = len(vecs['last_hidden_state'][0]) - 1
                    # extract rep of sent as last token
                    encs[wd][encoding].append(vecs['last_hidden_state'][0][idx_new].detach().numpy())

    return encs
