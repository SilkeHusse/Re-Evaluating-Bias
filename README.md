# Mind Your Bias
This repository contains the code and data for [Mind Your Bias: A Critical Review of Bias Detection Methods for Contextual Language Models](<link>). Corresponding supplementary is appended to the paper.

- [Set-up](#setup)
  - [Requirements](#reqs)
  - [Language Models](#models)
- [Data](#data)
  - [Reddit Comments](#reddit)
  - [Reduced and Simplified Word Sets](#mod)
- [User Manual](#manual)
  - [Generate Embeddings](#embeds)

<a name="setup"></a>
## Set-up

<a name="reqs"></a>
### Requirements
The whole code is executable with basic environments including common packages, e.g., `torch` or `transformers`, and does not require special attention. Solely, code snippets concerning working with ELMo demand a special environment which can be set up using [environment.yml](https://github.com/W4ngatang/sent-bias/blob/master/environment.yml). Please note that this file creates an environment only compatible for linux based systems. For macOS, use [environment_macOS.yml](https://github.com/SilkeHusse/Re-Evaluating-Bias/blob/main/methods/SEAT/environment_sentbiasMacOS.yml) and manually install all packages listed [here](https://github.com/SilkeHusse/Re-Evaluating-Bias/blob/main/methods/SEAT/environment_sentbiasMacOS.txt). Further, the environment required for ELMo is not suited for computations with OPT (respective code lines have to be (un)commented).

<a name="models"></a>
### Language Models
In the folder `models` create separate folders for each LM, named `elmo`, `bert`, `gpt2`, and `opt`, respectively. Additionally, create a single folders in the `bert` folder named `base-cased` and 'base-uncased`. This structure accounts for possible future enhancements to other LMs and specifically BERT versions. Save the required files for loading a LM from your local machine in the respective folder. 

<a name="data"></a>
## Data
We scrutinize eight bias tests (C1, C3, C6, C9, Dic, Occ, I1, I2). Respective datasets are further divided into a name or term category and are named correspondingly. An additional m behind a bias test's name indicates a modified dataset. Reduced and simplified datasets can be found in the `data/LPBS` folder. For the setting of template sentences, we distinguish between sentences containing single or double stimuli. Respective files `template_single` and `template_double` create final sentences.

<a name="reddit"></a>
### Reddit Comments
Providing all reddit data is infeasible and we recommend looking at the particular result files directly. If you decide to fetch the reddit data yourself, then in the subfolder `data/reddit` create a folder named `raw_data` and save all 12 bz2 files containing [reddit comments from 2014](https://files.pushshift.io/reddit/comments/). Additionally, create folders named `dict_files` and `processed_data` which are needed if `comment_*.py` code scripts are run. Those files have to be executed in the following order:
- `comment_fetch.py`: save relevant sentences from all 12 bz2 files in various pickle files (due to RAM restrictions)

> `dict_files/sent_dict_all_<month>_<idx>.pickle` <br />
- `comment_filter.py`: for each month filter sentences according to their single or double word affilitation and save them in a single pickle file

> `processed_data/sent_dict_single_<month>.pickle` <br />
> `processed_data/sent_dict_double_<month>.pickle` <br />
> `processed_data/sent_dict_single_simplified_<month>.pickle` <br />
> `processed_data/sent_dict_double_simplified_<month>.pickle` <br />
- `comment_sample_single.py`: from all 12 pickle files sample at most 10,000 sentences per stimuli and save them in a single pickle file (set respective parameter)

> `sent_dict_single.pickle` <br />
> `sent_dict_single_simplified.pickle` <br />
- `comment_sample_double.py`: from all 12 pickle files sample at most 10,000 sentences per stimuli and save them in a single pickle file (set respective parameter)

> `sent_dict_double.pickle` <br />
> `sent_dict_double_simplified.pickle` <br />

Alternatively, you can download all `processed_data` files and `sent_dict_*` files [here](<link>).
<a name="mod"></a>
### Reduced and Simplified Word Sets
To reproduce the results for the methods SEAT and CEAT in the case of reduced word sets, you will have to set the variable `reduced_wd_sets` in the respective `main.py` file to `True`. To reproduce the results for the methods SEAT and CEAT in the case of simplified word sets, you will have to set the variable `simpl_wd_sets` in the respective `main.py` file to `True`. Note that simplified word sets are only applicable if the context is `template`.

<a name="manual"></a>
## User Manual
To execute experiments, run `re_evaluating_bias.py` with specified parameters. You can find related code in the `methods` folder, organized by bias detection method. 

<a name="embeds"></a>
### Generate Embeddings
It is not feasible to generate embeddings in the context of reddit comments at runtime. Thus, for each method `generate_ebd_*.py` files exist which output the respective results in a csv file directly. For this, specific pickle files are required. They can be downloaded [here](<link>) and should be saved in the `data/reddit` folder. If you decide to fetch the reddit data yourself, please follow the instructions under **Set-up/Reddit Comments**. 
