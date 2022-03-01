""" Convenience functions for handling ELMo """
import os

from allennlp.commands.elmo import ElmoEmbedder

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load ELMo model and corresponding tokenizer from local files  """

    elmo = ElmoEmbedder(
        options_file=os.path.join(models_dir, 'elmo/options.json'),
        weight_file=os.path.join(models_dir, 'elmo/weights.hdf5'))
    return elmo

def encode(elmo, sents, stimulis, encoding):
    """ Function to encode sentences with ELMo """

    encs = {}
    for sent in sents:
        vec_seq = elmo.embed_sentence(sent.split())
        encoding_level = encoding[:4]
        if encoding_level == 'word': # here: no subword tokenization
            # determine idx of words in input sentence
            idx = None
            tokens = sent[:-1].split()
            for i, token in enumerate(tokens):
                if token.lower() in stimulis:
                    idx = i

            vec = vec_seq[:, idx] # extract rep of token of interest
            vec = vec.sum(axis=0)  # layer_combine_method = add
            encs[sent] = vec

        elif encoding_level == 'sent':
            vec = vec_seq.mean(axis=1) # extract rep of sent as average over all words
            vec = vec.sum(axis=0) # layer_combine_method = add
            encs[sent] = vec
    return encs
