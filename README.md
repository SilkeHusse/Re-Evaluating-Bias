# Re-Evaluating-Bias
This repository is the implementation of [(Re-)Evaluating Bias in Contextualized Word Embeddings](TODO link).
Supplementary?

- [Set-up](#setup)
  - [Requirements](#reqs)
  - [Language Models](#models)
  - [Reddit Data](#reddit)
  - [Shrunken Word Sets](#shrinking)
- [Generate Embeddings](#embeds)
- [Author Information](#author) 

<a name="setup"></a>
## Set-up

<a name="reqs"></a>
### Requirements
TODO provide own evironment.yml file --> descripe set-up

<a name="models"></a>
### Language Models
In the folder `models` create separate folders for each LM, named `elmo`, `bert`, and `gpt2`, respectively. Additionally, create a single folder in the `bert` folder named `base-cased`. This structure accounts for possible future enhancements to other LMs and specifically BERT versions. Save the required files for loading a LM from your local machine in the respective folder. 

<a name="reddit"></a>
### Reddit Data
If you decide to fetch the reddit data yourself, then in the subfolder `data/reddit` create a folder named `raw_data` and save all 12 bz2 files containing reddit comments from 2014 ([original dataset](https://files.pushshift.io/reddit/comments/). Additionally, create a folder named `dict_files` which is needed if `comment_*.py` code scripts are run. Those files have to be executed in the following order:
- `comment_fetch.py`: save relevant sentences from all 12 bz2 files in various pickle files (due to RAM restrictions)

> `dict_files/sent_dict_all_<month>_<idx>.pickle` <br />
- `comment_filter.py`: for each month filter the sentences according to their single or double word affilitation and save them in a single pickle file

> `sent_dict_single_<month>.pickle` and `sent_dict_double_<month>.pickle` <br />
- `comment_sample_single.py`: from all 12 pickle files sample at most 10,000 sentences per stimuli and save them in a single pickle file (another version with non-proportionate sampling can be found [here](https://www.kaggle.com/code/silkehusse/ceat-comment-sample-single))

> `sent_dict_single.pickle` <br />
- `comment_sample_double.py`: from all 12 pickle files sample at most 10,000 sentences per stimuli and save them in a single pickle file

> `sent_dict_double.pickle` <br />

<a name="shrinking"></a>
### Shrunken Word Sets
To reproduce the results for the methods SEAT and CEAT in the case of shrunken word sets, you will have to set the variable `shrunken_wd_sets` in the respective `main.py` file to `True`.

<a name="embeds"></a>
## Generate Embeddings
It is not feasible to generate embeddings in the context of reddit comments at runtime. Thus, for each method `generate_ebd_*.py` files exist which output the respective results in a csv file directly. For this, specific pickle files are required. They can be downloaded [here](https://www.kaggle.com/datasets/silkehusse/ceat-dict-files) and should be saved in the `data/reddit` folder. If you decide to fetch the reddit data yourself, please follow the instructions under **Set-up / reddit data**. 

<a name="author"></a>
## Author information
Silke Husse <br />
silke.husse@uni-konstanz.de
