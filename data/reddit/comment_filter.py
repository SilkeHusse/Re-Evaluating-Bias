""" filter relevant comments from dict_files """
import pickle
import os
import sys
import json
import itertools
import datetime
#import concurrent.futures

TEST_EXT = '.jsonl'
DICT_EXT = '.pickle'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.dirname(dirname)
dict_dir = os.path.join(dirname, 'dict_files')
thismodule = sys.modules[__name__] # for string to variable name conversion

all_tests = sorted([entry[:-len(TEST_EXT)]
                        for entry in os.listdir(data_dir)
                        if not entry.startswith('.') and entry.endswith('word' + TEST_EXT)])
all_dicts = sorted([entry[:-len(DICT_EXT)]
                        for entry in os.listdir(dict_dir)
                        if not entry.startswith('.') and entry.endswith(DICT_EXT)])

for test in all_tests:
    file_name = os.path.join(data_dir, '%s%s' % (test, TEST_EXT))
    file_data = json.load(open(file_name, 'r'))

    # for single word data sets
    for concept in ['targ1', 'targ2', 'attr1', 'attr2']:
        var_name = test[:-4] + concept
        setattr(thismodule, var_name.lower(), file_data[concept]['singular'])
        # this yields variables like eg c1_name_targ1, ..

    # for double word data sets
    test_name = test[:-5].lower()
    targ1 = file_data['targ1']['singular']
    targ2 = file_data['targ2']['singular']
    attr1 = file_data['attr1']['singular']
    attr2 = file_data['attr2']['singular']
    targ1attr1 = set(list(itertools.product(targ1, attr1)))
    targ1attr2 = set(list(itertools.product(targ1, attr2)))
    targ2attr1 = set(list(itertools.product(targ2, attr1)))
    targ2attr2 = set(list(itertools.product(targ2, attr2)))
    setattr(thismodule, test_name, set.union(targ1attr1, targ1attr2, targ2attr1, targ2attr2))
    # this yields variables like eg c1_name, ..

    # for double simplified word data sets
    #if test[:-5].lower() in ['c1_name', 'c3_term', 'c6_term', 'c9_name', 'occ_term']:
    if test_name == 'c1_name':
        targ1 = ['flower', 'flowers']
        targ2 = ['insect', 'insects']
    elif test_name == 'c3_term':
        targ1 = ['white']
        targ2 = ['black']
    elif test_name == 'c6_term' or test_name == 'occ_term':
        targ1 = ['he', 'men', 'boys']
        targ2 = ['she', 'women', 'girls']
    elif test_name == 'c9_name':
        targ1 = ['mental']
        targ2 = ['physical']

    targ1attr1 = set(list(itertools.product(targ1, attr1)))
    targ1attr2 = set(list(itertools.product(targ1, attr2)))
    targ2attr1 = set(list(itertools.product(targ2, attr1)))
    targ2attr2 = set(list(itertools.product(targ2, attr2)))
    var_name_simplified = test_name + '_simplified'
    setattr(thismodule, var_name_simplified, set.union(targ1attr1, targ1attr2, targ2attr1, targ2attr2))
    # this yields variables like eg c1_name_simplified, ..

# word list containing set of all single word target stimuli
wd_lst_single_targ = list(set(
    c1_name_targ1+c1_name_targ2+\
    c3_name_targ1+c3_name_targ2+\
    c3_term_targ1+c3_term_targ2+\
    c6_name_targ1+c6_name_targ2+\
    c6_term_targ1+c6_term_targ2+\
    c9_name_targ1+c9_name_targ2+\
    c9_term_targ1+c9_term_targ2+\
    c9_name_m_targ1+c9_name_m_targ2+\
    dis_term_targ1+dis_term_targ2+\
    dis_term_m_targ1+dis_term_m_targ2+\
    occ_name_targ1+occ_name_targ2+\
    occ_term_targ1+occ_term_targ2+\
    i1_name_targ1+i1_name_targ2+\
    i1_term_targ1+i1_term_targ2+\
    i2_name_targ1+i2_name_targ2+\
    i2_term_targ1+i2_term_targ2))

# word list containing set of all single word attribute stimuli
wd_lst_single_attr = list(set(
    c1_name_attr1+c1_name_attr2+\
    c3_name_attr1+c3_name_attr2+\
    c3_term_attr1+c3_term_attr2+\
    c6_name_attr1+c6_name_attr2+\
    c6_term_attr1+c6_term_attr2+\
    c9_name_attr1+c9_name_attr2+\
    c9_term_attr1+c9_term_attr2+\
    c9_name_m_attr1+c9_name_m_attr2+\
    dis_term_attr1+dis_term_attr2+\
    dis_term_m_attr1+dis_term_m_attr2+\
    occ_name_attr1+occ_name_attr2+\
    occ_term_attr1+occ_term_attr2+\
    i1_name_attr1+i1_name_attr2+\
    i1_term_attr1+i1_term_attr2+\
    i2_name_attr1+i2_name_attr2+\
    i2_term_attr1+i2_term_attr2))

# word list containing set of all single word stimuli
wd_lst_single = list(set.union(set(wd_lst_single_targ),set(wd_lst_single_attr)))
wd_lst_single_simplified = ['flower', 'flowers', 'insect', 'insects', 'white', 'black', 'he', 'men', 'boys', 'she', 'women', 'girls', 'mental', 'physical']

# word list containing set of all double word stimuli tuples
# in each tuple: (target,attribute)
wd_lst_double = list(set.union(c1_name, c3_name, c3_term, c6_name, c6_term, c9_name, c9_term, c9_name_m,
                               dis_term, dis_term_m, occ_name, occ_term, i1_name, i1_term, i2_name, i2_term))
wd_lst_double_simplified = list(set.union(c1_name_simplified, c3_term_simplified, c6_term_simplified,
                                          c9_name_simplified, occ_term_simplified))

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# load dict files and filter for relevant comments
def create_dicts(month):

    # create dict for double word stimuli where
    # key: stimuli ; value: dict with tuple as key and list of sents as value
    sent_dict_double = {i: {} for i in wd_lst_single_targ}
    for targ_stimuli in wd_lst_single_targ:
        stimuli_tuples = [wd_lst_double[i] for i in range(len(wd_lst_double)) if wd_lst_double[i][0] == targ_stimuli]
        sent_dict_double[targ_stimuli] = {i: [] for i in stimuli_tuples}

    # create dict for single word stimuli where
    # key: stimuli ; value: list of sents
    sent_dict_single = {i: [] for i in wd_lst_single}

    if month < 10:
        number_map = str(month) + '_'
        dict_file_names = [all_dicts[i] for i in range(len(all_dicts)) if all_dicts[i][14:16] == number_map]
    else:
        dict_file_names = [all_dicts[i] for i in range(len(all_dicts)) if all_dicts[i][14:16] == str(month)]

    for file in dict_file_names:
        file_name = 'dict_files/' + file + DICT_EXT
        dataset = pickle.load(open(file_name,'rb'))

        #now = datetime.datetime.now()
        #print(now.strftime("%Y-%m-%d %H:%M:%S"))
        #print(file_name)

        for key, value in dataset.items():
            if key in wd_lst_single_targ: # case: stimuli is target
                for comment in value:
                    appended = False
                    # step1: fill double word dicts
                    for key_tuple, value_tuple in sent_dict_double[key].items():
                        wwd = " " + key_tuple[1] + " "
                        # save comment if it contains attribute stimuli
                        if wwd in comment:
                            sent_dict_double[key][key_tuple].append(comment)
                            appended = True
                    # step2: fill single word dicts
                    if not appended:
                        sent_dict_single[key].append(comment)
            else: # case: stimuli is attribute
                # step2: fill single word dicts
                map_lst_targ = [wd_lst_double[i][0] for i in range(len(wd_lst_double)) if wd_lst_double[i][1] == key]
                for comment in value:
                    # check if sent contains target stimuli
                    if not any(wd in comment for wd in map_lst_targ):
                        sent_dict_single[key].append(comment)

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f'Done for month {month}')

    sent_dict_single_name = 'processed_data/sent_dict_single_' + str(month) + DICT_EXT
    pickle.dump(sent_dict_single, open(sent_dict_single_name, 'wb'))
    sent_dict_double_name = 'processed_data/sent_dict_double_' + str(month) + DICT_EXT
    pickle.dump(sent_dict_double, open(sent_dict_double_name, 'wb'))
    #return f'Finished month {month}'

# load dict files and filter for relevant comments (case: simplified)
def create_dicts_simplified(month):

    # create dict for double word stimuli where
    # key: stimuli ; value: dict with tuple as key and list of sents as value
    sent_dict_double = {i: {} for i in wd_lst_single_simplified}
    for targ_stimuli in wd_lst_single_simplified:
        stimuli_tuples = [wd_lst_double_simplified[i] for i in range(len(wd_lst_double_simplified)) if wd_lst_double_simplified[i][0] == targ_stimuli]
        sent_dict_double[targ_stimuli] = {i: [] for i in stimuli_tuples}

    # create dict for single word stimuli where
    # key: stimuli ; value: list of sents
    sent_dict_single = {i: [] for i in wd_lst_single_simplified}

    if month < 10:
        number_map = str(month) + '_'
        dict_file_names = [all_dicts[i] for i in range(len(all_dicts)) if all_dicts[i][14:16] == number_map]
    else:
        dict_file_names = [all_dicts[i] for i in range(len(all_dicts)) if all_dicts[i][14:16] == str(month)]

    for file in dict_file_names:
        file_name = 'dict_files/' + file + DICT_EXT
        dataset = pickle.load(open(file_name,'rb'))

        #now = datetime.datetime.now()
        #print(now.strftime("%Y-%m-%d %H:%M:%S"))
        #print(file_name)

        for key, value in dataset.items():
            for comment in value:
                appended = False
                # step1: fill double word dicts
                for key_tuple, value_tuple in sent_dict_double[key].items():
                    wwd = " " + key_tuple[1] + " "
                    # save comment if it contains attribute stimuli
                    if wwd in comment:
                        sent_dict_double[key][key_tuple].append(comment)
                        appended = True
                # step2: fill single word dicts
                if not appended:
                    sent_dict_single[key].append(comment)
            #else: # case: stimuli is attribute
            #    # step2: fill single word dicts
            #    map_lst_targ = [wd_lst_double[i][0] for i in range(len(wd_lst_double)) if wd_lst_double[i][1] == key]
            #    for comment in value:
            #        # check if sent contains target stimuli
            #        if not any(wd in comment for wd in map_lst_targ):
            #            sent_dict_single[key].append(comment)

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f'Done for month {month}')

    sent_dict_single_name = 'processed_data/sent_dict_single_simplified_' + str(month) + DICT_EXT
    pickle.dump(sent_dict_single, open(sent_dict_single_name, 'wb'))
    sent_dict_double_name = 'processed_data/sent_dict_double_simplified_' + str(month) + DICT_EXT
    pickle.dump(sent_dict_double, open(sent_dict_double_name, 'wb'))
    #return f'Finished month {month}'

#with concurrent.futures.ProcessPoolExecutor() as executor:
#    # if enough available RAM, then start and execute 12 processes
#    results = [executor.submit(create_dicts, i) for i in range(1, 13)]
#    for f in concurrent.futures.as_completed(results):
#        print(f.result())

for i in range(1,13):
    create_dicts(i)
    create_dicts_simplified(i)
