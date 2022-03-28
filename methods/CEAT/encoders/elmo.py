""" Convenience functions for handling ELMo """
import os

from allennlp.commands.elmo import ElmoEmbedder

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load ELMo model and corresponding tokenizer from local files """

    elmo = ElmoEmbedder(
        options_file=os.path.join(models_dir, 'elmo/options.json'),
        weight_file=os.path.join(models_dir, 'elmo/weights.hdf5'))
    return elmo

def encode(elmo, sents, stimuli, encodings):
    """ Function to encode sentences with ELMo """

    encs = {i:{'sent': [],
               'word-average': [],
               'word-start': [],
               'word-end': []} for i in stimuli}
    for wd in stimuli:
        for sent in sents[wd]:
            vec_seq = elmo.embed_sentence(sent.split())
            for encoding in encodings:
                encoding_level = encoding[:4]
                if encoding_level == 'word': # here: no subword tokenization
                    if len(wd.split()) > 1:
                        # determine idx of stimuli in input sentence
                        stimulus = [stimulus for stimulus in stimuli if stimulus in sent][0]
                        idx_start = sent[:-1].split().index(stimulus.split()[0])
                        # vector slicing excludes end idx
                        idx_end = idx_start + len(stimulus.split())
                        # extract reps of tokens of interest
                        vec = vec_seq[:, idx_start:idx_end, :]
                        # mean over all tokens of interest
                        vec = vec.mean(axis=1)
                    else:
                        # determine idx of stimulus in input sentence
                        idx = None
                        tokens = sent[:-1].split()
                        for i, token in enumerate(tokens):
                            if token in stimuli:
                                idx = i
                        # extract rep of token of interest
                        vec = vec_seq[:, idx, :]
                    encs[wd][encoding].append(vec.sum(axis=0)) # layer_combine_method = add

                elif encoding_level == 'sent':
                    # extract rep of sent as average over all words
                    vec = vec_seq.mean(axis=1)
                    encs[wd][encoding].append(vec.sum(axis=0)) # layer_combine_method = add
    return encs
