""" sample comments for single word dataset """
import pickle
import os
import sys
import json
import datetime
import random

random.seed(1111)
N = 10000

TEST_EXT = '.jsonl'
DICT_EXT = '.pickle'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.dirname(dirname)
thismodule = sys.modules[__name__] # for string to variable name conversion

all_tests = sorted([entry[:-len(TEST_EXT)]
                        for entry in os.listdir(data_dir)
                        if not entry.startswith('.') and entry.endswith('word' + TEST_EXT)])

for test in all_tests:
    file_name = os.path.join(data_dir, '%s%s' % (test, TEST_EXT))
    file_data = json.load(open(file_name, 'r'))

    # for single word data sets
    for concept in ['targ1', 'targ2', 'attr1', 'attr2']:
        var_name = test[:-4] + concept
        setattr(thismodule, var_name.lower(), file_data[concept]['singular'])
        # this yields variables like eg c1_name_targ1, ..

# word list containing set of all single word stimuli
wd_lst_single = list(set(
    c1_name_targ1+c1_name_targ2+c1_name_attr1+c1_name_attr2+\
    c3_name_targ1+c3_name_targ2+c3_name_attr1+c3_name_attr2+\
    c3_term_targ1+c3_term_targ2+c3_term_attr1+c3_term_attr2+\
    c6_name_targ1+c6_name_targ2+c6_name_attr1+c6_name_attr2+\
    c6_term_targ1+c6_term_targ2+c6_term_attr1+c6_term_attr2+\
    c9_name_targ1+c9_name_targ2+c9_name_attr1+c9_name_attr2+\
    c9_term_targ1+c9_term_targ2+c9_term_attr1+c9_term_attr2+\
    c9_name_m_targ1+c9_name_m_targ2+c9_name_m_attr1+c9_name_m_attr2+\
    dis_term_targ1+dis_term_targ2+dis_term_attr1+dis_term_attr2+\
    dis_term_m_targ1+dis_term_m_targ2+dis_term_m_attr1+dis_term_m_attr2+\
    occ_name_targ1+occ_name_targ2+occ_name_attr1+occ_name_attr2+\
    occ_term_targ1+occ_term_targ2+occ_term_attr1+occ_term_attr2+\
    i1_name_targ1+i1_name_targ2+i1_name_attr1+i1_name_attr2+\
    i1_term_targ1+i1_term_targ2+i1_term_attr1+i1_term_attr2+\
    i2_name_targ1+i2_name_targ2+i2_name_attr1+i2_name_attr2+\
    i2_term_targ1+i2_term_targ2+i2_term_attr1+i2_term_attr2))
wd_lst_single_simplified = ['flower', 'flowers', 'insect', 'insects', 'white', 'black', 'he', 'men', 'boys', 'she', 'women', 'girls', 'mental', 'physical']

# TODO set parameter, default = False
simplified = False

if simplified:
    data = wd_lst_single_simplified
else:
    data = wd_lst_single

sent_dict_prop = {i: [] for i in data}
sent_dict_total = {i: [] for i in data}
sent_dict_single = {i: [] for i in data}

# get proportion of comments per month and stimuli
for i in range(1,13):
    if simplified:
        dataset_name = 'processed_data/sent_dict_single_simplified_' + str(i) + '.pickle'
    else:
        dataset_name = 'processed_data/sent_dict_single_' + str(i) + '.pickle'

    dataset = pickle.load(open(dataset_name, 'rb'))
    for key, value in sent_dict_prop.items():
        sent_dict_prop[key].append((i,len(dataset[key])))

# get total number of comments per stimuli
for key, value in sent_dict_prop.items():
    n = 0
    for tuple in value:
        n += tuple[1]
    sent_dict_total[key] = n
#pickle.dump(sent_dict_total, open('sent_dict_single_n.pickle', 'wb'))

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print('Start sampling process...')

# fill and save final dict containing at most 10,000 comments per stimuli
for i in range(1,13):
    if simplified:
        dataset_name = 'processed_data/sent_dict_single_simplified_' + str(i) + '.pickle'
    else:
        dataset_name = 'processed_data/sent_dict_single_' + str(i) + '.pickle'

    dataset = pickle.load(open(dataset_name, 'rb'))
    for key, value in sent_dict_single.items():
        if sent_dict_total[key] > N:
            n = (sent_dict_prop[key][i - 1][1] / sent_dict_total[key]) * N # proportion to sample
            sents = random.sample(dataset[key], int(n))
            for sent in sents:
                sent_dict_single[key].append(sent)
        else:
            for sent in dataset[key]:
                sent_dict_single[key].append(sent)
    print(f'sent_dict_single: Sampled for month {i}')

if simplified:
    pickle.dump(sent_dict_single, open('sent_dict_single_simplified.pickle', 'wb'))
else:
    pickle.dump(sent_dict_single, open('sent_dict_single.pickle', 'wb'))
