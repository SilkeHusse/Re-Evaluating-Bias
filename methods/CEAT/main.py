import os
import json

from methods.CEAT import ceat
from methods.CEAT.encoders import elmo, bert, gpt2

TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'data')

def main(models, tests, encodings, contexts, evaluations):
    """ Main function of CEAT method """

    results = []
    for model in models:
        for test in tests:

            # TODO: indicate if shrunken word sets should be used
            shrunken_wd_sets = False
            # TODO: indicate if minimal word sets should be used
            minimal_wd_sets = False
            # load stimuli dataset
            if shrunken_wd_sets:
                try:
                    stimuli = json.load(open(os.path.join(data_dir, 'stimuli_logprob/shrunken_wd_sets/%s%s' % (test, TEST_EXT)), 'r'))
                except:
                    continue
            elif minimal_wd_sets:
                try:
                    stimuli = json.load(open(os.path.join(data_dir, 'stimuli_logprob/minimal_wd_sets/%s%s' % (test, TEST_EXT)), 'r'))
                except:
                    continue
            else:
                stimuli = json.load(open(os.path.join(data_dir, '%s%s' % (test, TEST_EXT)), 'r'))

            for measure in evaluations:

                if measure == 'cosine':
                    for context in contexts:

                        if context == 'template':
                            # load template sentences dataset
                            template_sents = json.load(open(
                                os.path.join(data_dir, '%s%s' % ('template_single', TEST_EXT)), 'r'))

                            if shrunken_wd_sets:
                                # define lists of bias tests sharing same specifications
                                specs_sp = ['C1_name_word', 'C3_name_word', 'C6_name_word', 'C6_term_word']
                                specs_s = ['Occ_name_word', 'Occ_term_word', 'C9_term_word', 'C9_name_word', 'C9m_name_word']

                                if test in specs_sp:
                                    stimuli_targ1 = stimuli['targ1']['examples_singular']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                    + stimuli['attr1']['examples_plural']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                    + stimuli['attr2']['examples_plural']
                                elif test in specs_s:
                                    stimuli_targ1 = stimuli['targ1']['examples_singular']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular']
                                else:
                                    raise ValueError("Shrunken bias test %s not found!" % test)
                            elif minimal_wd_sets:
                                # define lists of bias tests sharing same specifications
                                specs_sp = ['C1_name_word', 'C3_term_word', 'C6_term_word']
                                specs_s = ['C9_name_word', 'C9m_name_word']

                                if test in specs_sp:
                                    stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                    + stimuli['targ1']['examples_plural']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                    + stimuli['targ2']['examples_plural']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                    + stimuli['attr1']['examples_plural']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                    + stimuli['attr2']['examples_plural']
                                elif test in specs_s:
                                    stimuli_targ1 = stimuli['targ1']['examples_singular']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular']
                                elif test == 'Occ_term_word':
                                    stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                    + stimuli['targ1']['examples_plural']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                    + stimuli['targ2']['examples_plural']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular']
                                else:
                                    raise ValueError("Minimal bias test %s not found!" % test)
                            else:
                                # define lists of bias tests sharing same specifications
                                specs_sp = ['C1_name_word', 'C3_term_word', 'C6_term_word']
                                specs_s = ['IBD_term_word', 'EIBD_term_word', 'IBD_name_word', 'EIBD_name_word',
                                           'Dis_term_word', 'Dism_term_word', 'Occ_name_word', 'C9_term_word',
                                           'C9_name_word', 'C9m_name_word']

                                if test in specs_sp:
                                    stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                    + stimuli['targ1']['examples_plural']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                    + stimuli['targ2']['examples_plural']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                    + stimuli['attr1']['examples_plural']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                    + stimuli['attr2']['examples_plural']
                                elif test in specs_s:
                                    stimuli_targ1 = stimuli['targ1']['examples_singular']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular']
                                elif test == 'C3_name_word' or test == 'C6_name_word':
                                    stimuli_targ1 = stimuli['targ1']['examples_singular']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                    + stimuli['attr1']['examples_plural']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                    + stimuli['attr2']['examples_plural']
                                elif test == 'Occ_term_word':
                                    stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                    + stimuli['targ1']['examples_plural']
                                    stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                    + stimuli['targ2']['examples_plural']
                                    stimuli_attr1 = stimuli['attr1']['examples_singular']
                                    stimuli_attr2 = stimuli['attr2']['examples_singular']
                                else:
                                    raise ValueError("Bias test %s not found!" % test)

                            sents_targ1, sents_targ2 = {i:[] for i in stimuli_targ1}, {i:[] for i in stimuli_targ2}
                            sents_attr1, sents_attr2 = {i:[] for i in stimuli_attr1}, {i:[] for i in stimuli_attr2}

                            # for each bias test
                            # create sents by replacing target and attribute words in template sentences
                            if shrunken_wd_sets:
                                if test == 'C1_name_word':
                                    for sent in template_sents['targ']['singular_thing']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C3_name_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C6_name_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C6_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C9_name_word' or test == 'C9m_name_word' or test == 'C9_term_word':
                                    for sent in template_sents['targ']['singular_thing']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_time']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'Occ_name_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                elif test == 'Occ_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                            elif minimal_wd_sets:
                                if test == 'C1_name_word':
                                    for sent in template_sents['targ']['singular_thing']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                    for sent in template_sents['targ']['plural_thing']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C3_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' person'))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' person'))
                                    for sent in template_sents['targ']['plural_person']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus + ' people'))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus + ' people'))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C6_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['targ']['plural_person']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C9_name_word' or test == 'C9m_name_word':
                                    for sent in template_sents['targ']['singular_thing']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' disease'))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' disease'))
                                    for sent in template_sents['attr']['singular_time']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'Occ_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['targ']['plural_person']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                            else:
                                if test == 'C1_name_word':
                                    for sent in template_sents['targ']['singular_thing']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                    for sent in template_sents['targ']['plural_thing']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C3_name_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C3_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                    for sent in template_sents['targ']['plural_person']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C6_name_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C6_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['targ']['plural_person']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                    for sent in template_sents['attr']['plural_basic']:
                                        for stimulus in stimuli['attr1']['examples_plural']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_plural']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'C9_name_word' or test == 'C9m_name_word' or test == 'C9_term_word':
                                    for sent in template_sents['targ']['singular_thing']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_time']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'Occ_name_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                elif test == 'Occ_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['targ']['plural_person']:
                                        for stimulus in stimuli['targ1']['examples_plural']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_plural']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_basic']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                                elif test == 'Dis_term_word' or test == 'Dism_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_time']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'IBD_name_word' or test == 'EIBD_name_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', stimulus))
                                    for sent in template_sents['attr']['singular_time']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))
                                elif test == 'IBD_term_word' or test == 'EIBD_term_word':
                                    for sent in template_sents['targ']['singular_person']:
                                        for stimulus in stimuli['targ1']['examples_singular']:
                                            sents_targ1[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                        for stimulus in stimuli['targ2']['examples_singular']:
                                            sents_targ2[stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                                    for sent in template_sents['attr']['singular_time']:
                                        for stimulus in stimuli['attr1']['examples_singular']:
                                            sents_attr1[stimulus].append(sent.replace('AAA', stimulus))
                                        for stimulus in stimuli['attr2']['examples_singular']:
                                            sents_attr2[stimulus].append(sent.replace('AAA', stimulus))

                            if model == 'elmo':
                                model_loaded = elmo.load_model()
                                encs_targ1 = elmo.encode(model_loaded,
                                                         sents_targ1, stimuli_targ1, encodings)
                                encs_targ2 = elmo.encode(model_loaded,
                                                         sents_targ2, stimuli_targ2, encodings)
                                encs_attr1 = elmo.encode(model_loaded,
                                                         sents_attr1, stimuli_attr1, encodings)
                                encs_attr2 = elmo.encode(model_loaded,
                                                         sents_attr2, stimuli_attr2, encodings)
                            elif model == 'bert':
                                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = bert.load_model()
                                encs_targ1 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ1, stimuli_targ1, encodings)
                                encs_targ2 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ2, stimuli_targ2, encodings)
                                encs_attr1 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr1, stimuli_attr1, encodings)
                                encs_attr2 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr2, stimuli_attr2, encodings)
                            elif model == 'gpt2':
                                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = gpt2.load_model()
                                encs_targ1 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ1, stimuli_targ1, encodings)
                                encs_targ2 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ2, stimuli_targ2, encodings)
                                encs_attr1 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr1, stimuli_attr1, encodings)
                                encs_attr2 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr2, stimuli_attr2, encodings)
                            else:
                                raise ValueError("Model %s not found!" % model)

                        elif context == 'reddit':
                            print(f'For context {context} no results can be generated at runtime and thus is skipped.')
                            print(f'Please see the results folder directly or execute a respective generate_ebd_* file.')
                            break
                        else:
                            raise ValueError("Context %s not found!" % context)

                        encs = {}
                        encs["targ1"] = {'encs': encs_targ1}
                        encs["targ2"] = {'encs': encs_targ2}
                        encs["attr1"] = {'encs': encs_attr1}
                        encs["attr2"] = {'encs': encs_attr2}

                        for encoding in encodings:
                            # default parameter: N = 10,000
                            esize, pval = ceat.ceat_meta(encs, encoding)
                            results.append(dict(
                                method='CEAT',
                                test=test,
                                model=model,
                                evaluation_measure=measure,
                                context=context,
                                encoding_level=encoding,
                                p_value=pval,
                                effect_size=esize))

                # TODO
                elif measure == 'prob':
                    pass

    return results
