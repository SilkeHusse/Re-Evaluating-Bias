""" Convenience function for handling GPT2 """
import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2TokenizerFast

dirname = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dirname))), 'models')

def load_model():
    """ Load GPT2 model and corresponding tokenizers """

    tokenizer = GPT2Tokenizer.from_pretrained(models_dir + '/gpt2/')
    # additional 'Fast' GPT2 tokenizer for subword tokenization ID mapping
    subword_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # set parameter pad_token_id to eos token for word sequence generation
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    model.eval()

    return model, tokenizer, subword_tokenizer
