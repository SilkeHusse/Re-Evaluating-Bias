""" collect all relevant comments from reddit dataset """
import bz2
import string
import datetime
import json
import pickle
import os
import sys
import concurrent.futures

TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.dirname(dirname)
thismodule = sys.modules[__name__] # for string to variable name conversion

all_tests = sorted([entry[:-len(TEST_EXT)]
                        for entry in os.listdir(data_dir)
                        if not entry.startswith('.') and entry.endswith('word' + TEST_EXT)])

for test in all_tests:
    file_name = os.path.join(data_dir, '%s%s' % (test, TEST_EXT))
    file_data = json.load(open(file_name, 'r'))
    # for each stimuli
    for concept in ['targ1', 'targ2', 'attr1', 'attr2']:
        var_name = test[:-4] + concept
        setattr(thismodule, var_name.lower(), file_data[concept]['examples_singular'])
        # this yields variables like eg c1_name_targ1, ..

# word list containing set of all stimuli
wd_lst = list(set(
    c1_name_targ1+c1_name_targ2+c1_name_attr1+c1_name_attr2+\
    c3_name_targ1+c3_name_targ2+c3_name_attr1+c3_name_attr2+\
    c3_term_targ1+c3_term_targ2+c3_term_attr1+c3_term_attr2+\
    c6_name_targ1+c6_name_targ2+c6_name_attr1+c6_name_attr2+\
    c6_term_targ1+c6_term_targ2+c6_term_attr1+c6_term_attr2+\
    c9_name_targ1+c9_name_targ2+c9_name_attr1+c9_name_attr2+\
    c9_term_targ1+c9_term_targ2+c9_term_attr1+c9_term_attr2+\
    c9m_name_targ1+c9m_name_targ2+c9m_name_attr1+c9m_name_attr2+\
    dis_term_targ1+dis_term_targ2+dis_term_attr1+dis_term_attr2+\
    dism_term_targ1+dism_term_targ2+dism_term_attr1+dism_term_attr2+\
    occ_name_targ1+occ_name_targ2+occ_name_attr1+occ_name_attr2+\
    occ_term_targ1+occ_term_targ2+occ_term_attr1+occ_term_attr2+\
    ibd_name_targ1+ibd_name_targ2+ibd_name_attr1+ibd_name_attr2+\
    ibd_term_targ1+ibd_term_targ2+ibd_term_attr1+ibd_term_attr2+\
    eibd_name_targ1+eibd_name_targ2+eibd_name_attr1+eibd_name_attr2+\
    eibd_term_targ1+eibd_term_targ2+eibd_term_attr1+eibd_term_attr2))

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# for text pre-processing
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def create_dict(idx_file):
    count_dict_all = {i:0 for i in wd_lst}
    sent_dict_all = {i:[] for i in wd_lst}
    idx_dict = 0

    if idx_file<10:
        file_path = 'raw_data/RC_2014'+'-0' + str(idx_file) + '.bz2'
    else:
        file_path = 'raw_data/RC_2014'+'-' + str(idx_file) + '.bz2'
    #print(file_path)

    latest_timestamp = datetime.datetime.now()
    #print(latest_timestamp.strftime("%Y-%m-%d %H:%M:%S"))

    with bz2.open(file_path,mode='rt',encoding='utf-8') as f:
        for idx,line in enumerate(f):

            current_timestamp = datetime.datetime.now()
            time_delta = current_timestamp-latest_timestamp
            # check if 30 min passed since last dict save, adapt to own hardware restrictions
            # here specifications for 16GB RAM
            if time_delta.total_seconds() > 1800:
                # save current dict to pickle file
                sent_dict_name = 'dict_files/sent_dict_all_' + str(idx_file) + '_' + str(idx_dict) + '.pickle'
                pickle.dump(sent_dict_all, open(sent_dict_name, 'wb'))
                idx_dict += 1 # increase idx for next dict save
                sent_dict_all = {i: [] for i in wd_lst} # reset dict
                latest_timestamp = current_timestamp # reset latest timestamp

            comment_line = json.loads(line)
            comment = comment_line["body"]
            if comment != '[deleted]' and comment != '[removed]':
                comment = comment.replace("&gt;"," ")
                comment = comment.replace("&amp;"," ")
                comment = comment.replace("&lt;"," ")
                comment = comment.replace("&quot;"," ")
                comment = comment.replace("&apos;"," ")
                comment = comment.translate(translator)
                for wd in wd_lst:
                    wwd = " "+wd+" "
                    # save all comments which contain stimuli
                    if wwd in comment:
                        count_dict_all[wd] += 1
                        sent_dict_all[wd].append(comment)

    # save last dict to pickle file
    sent_dict_name = 'dict_files/sent_dict_all_' + str(idx_file) + '_' + str(idx_dict) + '.pickle'
    pickle.dump(sent_dict_all, open(sent_dict_name, 'wb'))
    return f'Done for month {idx_file}'

with concurrent.futures.ProcessPoolExecutor() as executor:
    # start and execute 12 processes
    results = [executor.submit(create_dict, i) for i in range(1, 13)]
    for f in concurrent.futures.as_completed(results):
        print(f.result())

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
