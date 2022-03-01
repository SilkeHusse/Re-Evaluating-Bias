import os
import random

from methods.SEAT import data, weat
from methods.SEAT.encoders import bert, elmo, gpt2

random.seed(1111)
TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'data')

def main(models, tests, encodings, contexts, evaluations, parametric):
    """ Main function of SEAT method"""

    results = []
    for model in models:
        for test in tests:
            # load single word dataset
            single_words = data.load_json(os.path.join(data_dir, '%s%s' % (test, TEST_EXT)))
            for measure in evaluations:

                if measure == 'cosine':
                    for context in contexts:
                        sents_targ1, sents_targ2 = [], []
                        sents_attr1, sents_attr2 = [], []
                        stimuli_targ1, stimuli_targ2 = [], []
                        stimuli_attr1, stimuli_attr2 = [], []

                        if context == 'template':
                            # load template sentences dataset
                            template_sents = data.load_json(os.path.join(data_dir, '%s%s' % ('template_single', TEST_EXT)))

                            # for each bias test
                            # - extract stimuli from single word dataset
                            # - create sents by replacing target and attribute words in template sentences
                            same_characs = ['Dis_term_word', 'Dism_term_word', 'IBD_name_word', 'IBD_term_word','EIBD_name_word', 'EIBD_term_word']
                            if test == 'C1_name_word':
                                stimuli_targ1 = single_words['targ1']['examples_singular'] + single_words['targ1']['examples_plural']
                                stimuli_targ2 = single_words['targ2']['examples_singular'] + single_words['targ2']['examples_plural']
                                stimuli_attr1 = single_words['attr1']['examples_singular'] + single_words['attr1']['examples_plural']
                                stimuli_attr2 = single_words['attr2']['examples_singular'] + single_words['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_thing']:
                                    for single_word in single_words['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['targ']['plural_thing']:
                                    for single_word in single_words['targ1']['examples_plural']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_plural']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['attr']['singular_basic']:
                                    for single_word in single_words['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                                for sent in template_sents['attr']['plural_basic']:
                                    for single_word in single_words['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                            elif test == 'C3_name_word' or test == 'C6_name_word':
                                stimuli_targ1 = single_words['targ1']['examples_singular']
                                stimuli_targ2 = single_words['targ2']['examples_singular']
                                stimuli_attr1 = single_words['attr1']['examples_singular'] + single_words['attr1']['examples_plural']
                                stimuli_attr2 = single_words['attr2']['examples_singular'] + single_words['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_person']:
                                    for single_word in single_words['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['attr']['singular_basic']:
                                    for single_word in single_words['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                                for sent in template_sents['attr']['plural_basic']:
                                    for single_word in single_words['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                            elif test == 'C3_term_word' or test == 'C6_term_word':
                                stimuli_targ1 = single_words['targ1']['examples_singular'] + single_words['targ1']['examples_plural']
                                stimuli_targ2 = single_words['targ2']['examples_singular'] + single_words['targ2']['examples_plural']
                                stimuli_attr1 = single_words['attr1']['examples_singular'] + single_words['attr1']['examples_plural']
                                stimuli_attr2 = single_words['attr2']['examples_singular'] + single_words['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_person']:
                                    for single_word in single_words['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['targ']['plural_person']:
                                    for single_word in single_words['targ1']['examples_plural']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_plural']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['attr']['singular_basic']:
                                    for single_word in single_words['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                                for sent in template_sents['attr']['plural_basic']:
                                    for single_word in single_words['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                            elif test == 'C9_name_word' or test == 'C9m_name_word' or 'C9_term_word':
                                stimuli_targ1 = single_words['targ1']['examples_singular']
                                stimuli_targ2 = single_words['targ2']['examples_singular']
                                stimuli_attr1 = single_words['attr1']['examples_singular']
                                stimuli_attr2 = single_words['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_thing']:
                                    for single_word in single_words['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['attr']['singular_time']:
                                    for single_word in single_words['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                            elif test == 'Occ_name_word':
                                stimuli_targ1 = single_words['targ1']['examples_singular']
                                stimuli_targ2 = single_words['targ2']['examples_singular']
                                stimuli_attr1 = single_words['attr1']['examples_singular']
                                stimuli_attr2 = single_words['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for single_word in single_words['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['attr']['singular_basic']:
                                    for single_word in single_words['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                            elif test == 'Occ_term_word':
                                stimuli_targ1 = single_words['targ1']['examples_singular'] + single_words['targ1']['examples_plural']
                                stimuli_targ2 = single_words['targ2']['examples_singular'] + single_words['targ2']['examples_plural']
                                stimuli_attr1 = single_words['attr1']['examples_singular']
                                stimuli_attr2 = single_words['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for single_word in single_words['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['targ']['plural_person']:
                                    for single_word in single_words['targ1']['examples_plural']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_plural']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['attr']['singular_basic']:
                                    for single_word in single_words['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                            elif test in same_characs:
                                stimuli_targ1 = single_words['targ1']['examples_singular']
                                stimuli_targ2 = single_words['targ2']['examples_singular']
                                stimuli_attr1 = single_words['attr1']['examples_singular']
                                stimuli_attr2 = single_words['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for single_word in single_words['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', single_word))
                                    for single_word in single_words['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', single_word))
                                for sent in template_sents['attr']['singular_time']:
                                    for single_word in single_words['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', single_word))
                                    for single_word in single_words['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', single_word))
                            else:
                                raise ValueError("Bias test %s not found!" % test)

                            # target sets have to be of equal size
                            if not len(sents_targ1) == len(sents_targ2):
                                min_n = min([len(sents_targ1), len(sents_targ2)])
                                # randomly sample min number of sents for both word sets
                                if not len(sents_targ1) == min_n:
                                    sents_targ1 = random.sample(sents_targ1, min_n)
                                else:
                                    sents_targ2 = random.sample(sents_targ2, min_n)

                        # TODO context == reddit
                        elif context == 'reddit':
                            pass
                        else:
                            raise ValueError("Context %s not found!" % context)

                        for encoding in encodings:
                            if model == 'elmo':
                                model_elmo = elmo.load_model()
                                encs_targ1 = elmo.encode(model_elmo, sents_targ1, stimuli_targ1, encoding)
                                encs_targ2 = elmo.encode(model_elmo, sents_targ2, stimuli_targ2, encoding)
                                encs_attr1 = elmo.encode(model_elmo, sents_attr1, stimuli_attr1, encoding)
                                encs_attr2 = elmo.encode(model_elmo, sents_attr2, stimuli_attr2, encoding)
                            elif model == 'bert':
                                model_bert, tokenizer_bert, subword_tokenizer_bert = bert.load_model()
                                encs_targ1 = bert.encode(model_bert, tokenizer_bert, subword_tokenizer_bert, sents_targ1, stimuli_targ1, encoding)
                                encs_targ2 = bert.encode(model_bert, tokenizer_bert, subword_tokenizer_bert, sents_targ2, stimuli_targ2, encoding)
                                encs_attr1 = bert.encode(model_bert, tokenizer_bert, subword_tokenizer_bert, sents_attr1, stimuli_attr1, encoding)
                                encs_attr2 = bert.encode(model_bert, tokenizer_bert, subword_tokenizer_bert, sents_attr2, stimuli_attr2, encoding)
                            elif model == 'gpt2':
                                model_gpt2, tokenizer_gpt2, subword_tokenizer_gpt2 = gpt2.load_model()
                                encs_targ1 = gpt2.encode(model_gpt2, tokenizer_gpt2, subword_tokenizer_gpt2, sents_targ1, stimuli_targ1, encoding)
                                encs_targ2 = gpt2.encode(model_gpt2, tokenizer_gpt2, subword_tokenizer_gpt2, sents_targ2, stimuli_targ2, encoding)
                                encs_attr1 = gpt2.encode(model_gpt2, tokenizer_gpt2, subword_tokenizer_gpt2, sents_attr1, stimuli_attr1, encoding)
                                encs_attr2 = gpt2.encode(model_gpt2, tokenizer_gpt2, subword_tokenizer_gpt2, sents_attr2, stimuli_attr2, encoding)
                            else:
                                raise ValueError("Model %s not found!" % model)

                            encs = {}
                            encs["targ1"] = {'concept': single_words['targ1']['concept'], 'encs': encs_targ1}
                            encs["targ2"] = {'concept': single_words['targ2']['concept'], 'encs': encs_targ2}
                            encs["attr1"] = {'concept': single_words['attr1']['concept'], 'encs': encs_attr1}
                            encs["attr2"] = {'concept': single_words['attr2']['concept'], 'encs': encs_attr2}

                            # default parameter: n_samples = 100,000
                            esize, pval = weat.run_test(encs, parametric)
                            results.append(dict(
                                method='s-SEAT',
                                test=test,
                                model=model,
                                evaluation_measure=measure,
                                context=context,
                                encoding_level=encoding,
                                p_value=pval,
                                effect_size=esize,
                                num_targ1=len(encs['targ1']['encs']),
                                num_targ2=len(encs['targ2']['encs']),
                                num_attr1=len(encs['attr1']['encs']),
                                num_attr2=len(encs['attr2']['encs'])))

                # TODO measure == prob
                elif measure == 'prob':
                    pass

    return results