""" Convenience function for loading BERT """
import os

from transformers import BertTokenizer, BertForMaskedLM
from transformers import BertTokenizerFast

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load BERT model (bbu) and corresponding tokenizers """

    tokenizer = BertTokenizer.from_pretrained(models_dir + '/bert/base-uncased/')
    # additional 'Fast' BERT tokenizer for subword tokenization ID mapping
    subword_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained(models_dir + '/bert/base-uncased/')
    model.eval()

    return model, tokenizer, subword_tokenizer
