# code snippet to download LMs (weights & configs) on local machine

# BERT (https://huggingface.co/docs/transformers/model_doc/bert)
# https://huggingface.co/bert-base-cased

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")
tokenizer.save_pretrained('models/bert/')
model.save_pretrained('models/bert/')

# GPT-2 (https://huggingface.co/docs/transformers/model_doc/gpt2)
# https://huggingface.co/gpt2

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
tokenizer.save_pretrained('models/gpt-2/')
model.save_pretrained('models/gpt-2/')

# ELMo (https://allenai.org/allennlp/software/elmo)
# http://docs.allennlp.org/v0.9.0/api/allennlp.commands.elmo.html

from urllib import request
options_url = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weights_url = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
options_file = 'models/elmo/options.json'
weights_file = 'models/elmo/weights.hdf5'
request.urlretrieve(options_url, options_file)
request.urlretrieve(weights_url, weights_file)