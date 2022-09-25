# code snippet to download LMs (weights & configs) on local machine

# BERT (https://huggingface.co/docs/transformers/model_doc/bert)
# https://huggingface.co/bert-base-uncased

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained('bert/base-uncased/')
model.save_pretrained('bert/base-uncased/')

# BERT (https://huggingface.co/docs/transformers/model_doc/bert)
# https://huggingface.co/bert-base-cased

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")
tokenizer.save_pretrained('bert/base-cased/')
model.save_pretrained('bert/bert-base-cased/')

# GPT-2 (https://huggingface.co/docs/transformers/model_doc/gpt2)
# https://huggingface.co/gpt2

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
tokenizer.save_pretrained('gpt2/')
model.save_pretrained('gpt2/')

# ELMo (https://allenai.org/allennlp/software/elmo)
# http://docs.allennlp.org/v0.9.0/api/allennlp.commands.elmo.html

from urllib import request
options_url = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weights_url = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
options_file = 'elmo/options.json'
weights_file = 'elmo/weights.hdf5'
request.urlretrieve(options_url, options_file)
request.urlretrieve(weights_url, weights_file)

# OPT (https://huggingface.co/docs/transformers/model_doc/opt)
# https://huggingface.co/facebook/opt-125m

from transformers import GPT2Tokenizer, OPTModel
tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-125m')
model = OPTModel.from_pretrained('facebook/opt-125m')
tokenizer.save_pretrained('opt/')
model.save_pretrained('opt/')

# BLOOM (https://huggingface.co/docs/transformers/model_doc/bloom)
# https://huggingface.co/bigscience/bloom-560m

from transformers import BloomTokenizerFast, BloomModel
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomModel.from_pretrained("bigscience/bloom-560m")
tokenizer.save_pretrained('bloom/')
model.save_pretrained('bloom/')
