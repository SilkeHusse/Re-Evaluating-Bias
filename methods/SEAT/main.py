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
        if context != 'reddit': # only load models if required
            if model == 'elmo':
                model_loaded = elmo.load_model()
                tokenizer_loaded, subword_tokenizer_loaded = None, None
            elif model == 'bert':
                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = bert.load_model()
            elif model == 'gpt2':
                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = gpt2.load_model()
            else:
                raise ValueError("Model %s not found!" % model)

        for test in tests:
            # load stimuli dataset
            stimuli = data.load_json(os.path.join(data_dir, '%s%s' % (test, TEST_EXT)))
            for measure in evaluations:

                if measure == 'cosine':
                    for context in contexts:
                        sents_targ1, sents_targ2 = [], []
                        sents_attr1, sents_attr2 = [], []
                        stimuli_targ1, stimuli_targ2 = [], []
                        stimuli_attr1, stimuli_attr2 = [], []

                        if context == 'template':
                            # load template sentences dataset
                            template_sents = data.load_json(os.path.join(data_dir, '%s%s' %
                                                                         ('template_single', TEST_EXT)))

                            # for each bias test
                            # - extract stimuli from stimuli dataset
                            # - create sents by replacing target and attribute words in template sentences
                            multiple_targ = False # bool indicating if targ stimuli consist of multiple words
                            multiple_attr = False # bool indicating if attr stimuli consist of multiple words
                            if test == 'C1_name_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                + stimuli['targ1']['examples_plural']
                                stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                + stimuli['targ2']['examples_plural']
                                stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                + stimuli['attr1']['examples_plural']
                                stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                + stimuli['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_thing']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', 'a ' + stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', 'a ' + stimulus))
                                for sent in template_sents['targ']['plural_thing']:
                                    for stimulus in stimuli['targ1']['examples_plural']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_plural']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_basic']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                                for sent in template_sents['attr']['plural_basic']:
                                    for stimulus in stimuli['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'C3_name_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular']
                                stimuli_targ2 = stimuli['targ2']['examples_singular']
                                stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                + stimuli['attr1']['examples_plural']
                                stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                + stimuli['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_basic']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                                for sent in template_sents['attr']['plural_basic']:
                                    for stimulus in stimuli['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'C3_term_word':
                                multiple_targ = True
                                stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                + stimuli['targ1']['examples_plural']
                                stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                + stimuli['targ2']['examples_plural']
                                stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                + stimuli['attr1']['examples_plural']
                                stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                + stimuli['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', 'a ' + stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', 'a ' + stimulus))
                                for sent in template_sents['targ']['plural_person']:
                                    for stimulus in stimuli['targ1']['examples_plural']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_plural']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_basic']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                                for sent in template_sents['attr']['plural_basic']:
                                    for stimulus in stimuli['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'C6_name_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular']
                                stimuli_targ2 = stimuli['targ2']['examples_singular']
                                stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                + stimuli['attr1']['examples_plural']
                                stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                + stimuli['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_basic']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', 'a ' + stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', 'a ' + stimulus))
                                for sent in template_sents['attr']['plural_basic']:
                                    for stimulus in stimuli['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'C6_term_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                + stimuli['targ1']['examples_plural']
                                stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                + stimuli['targ2']['examples_plural']
                                stimuli_attr1 = stimuli['attr1']['examples_singular'] \
                                                + stimuli['attr1']['examples_plural']
                                stimuli_attr2 = stimuli['attr2']['examples_singular'] \
                                                + stimuli['attr2']['examples_plural']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['targ']['plural_person']:
                                    for stimulus in stimuli['targ1']['examples_plural']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_plural']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_basic']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', 'a ' + stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', 'a ' + stimulus))
                                for sent in template_sents['attr']['plural_basic']:
                                    for stimulus in stimuli['attr1']['examples_plural']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_plural']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'C9_name_word' or test == 'C9m_name_word' or test == 'C9_term_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular']
                                stimuli_targ2 = stimuli['targ2']['examples_singular']
                                stimuli_attr1 = stimuli['attr1']['examples_singular']
                                stimuli_attr2 = stimuli['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_thing']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_time']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'Occ_name_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular']
                                stimuli_targ2 = stimuli['targ2']['examples_singular']
                                stimuli_attr1 = stimuli['attr1']['examples_singular']
                                stimuli_attr2 = stimuli['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_basic']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', 'a ' + stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', 'a ' + stimulus))
                            elif test == 'Occ_term_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular'] \
                                                + stimuli['targ1']['examples_plural']
                                stimuli_targ2 = stimuli['targ2']['examples_singular'] \
                                                + stimuli['targ2']['examples_plural']
                                stimuli_attr1 = stimuli['attr1']['examples_singular']
                                stimuli_attr2 = stimuli['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['targ']['plural_person']:
                                    for stimulus in stimuli['targ1']['examples_plural']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_plural']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_basic']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', 'a ' + stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', 'a ' + stimulus))
                            elif test == 'Dis_term_word' or test == 'Dism_term_word':
                                multiple_targ = True
                                stimuli_targ1 = stimuli['targ1']['examples_singular']
                                stimuli_targ2 = stimuli['targ2']['examples_singular']
                                stimuli_attr1 = stimuli['attr1']['examples_singular']
                                stimuli_attr2 = stimuli['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_time']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'IBD_name_word' or test == 'EIBD_name_word':
                                stimuli_targ1 = stimuli['targ1']['examples_singular']
                                stimuli_targ2 = stimuli['targ2']['examples_singular']
                                stimuli_attr1 = stimuli['attr1']['examples_singular']
                                stimuli_attr2 = stimuli['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', stimulus))
                                for sent in template_sents['attr']['singular_time']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
                            elif test == 'IBD_term_word' or test == 'EIBD_term_word':
                                multiple_targ = True
                                stimuli_targ1 = stimuli['targ1']['examples_singular']
                                stimuli_targ2 = stimuli['targ2']['examples_singular']
                                stimuli_attr1 = stimuli['attr1']['examples_singular']
                                stimuli_attr2 = stimuli['attr2']['examples_singular']
                                for sent in template_sents['targ']['singular_person']:
                                    for stimulus in stimuli['targ1']['examples_singular']:
                                        sents_targ1.append(sent.replace('TTT', 'a ' + stimulus))
                                    for stimulus in stimuli['targ2']['examples_singular']:
                                        sents_targ2.append(sent.replace('TTT', 'a ' + stimulus))
                                for sent in template_sents['attr']['singular_time']:
                                    for stimulus in stimuli['attr1']['examples_singular']:
                                        sents_attr1.append(sent.replace('AAA', stimulus))
                                    for stimulus in stimuli['attr2']['examples_singular']:
                                        sents_attr2.append(sent.replace('AAA', stimulus))
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

                        elif context == 'reddit':
                            print(f'For context {context} no results can be generated at runtime and thus is skipped.')
                            print(
                                f'Please see the results folder directly or execute a respective generate_ebd_* file.')
                            break
                        else:
                            raise ValueError("Context %s not found!" % context)

                        # TODO: check if multiple_targ and multiple_attr variables were assigned as booleans

                        for encoding in encodings:
                            if model == 'elmo':
                                encs_targ1 = elmo.encode(model_loaded,
                                                         sents_targ1, stimuli_targ1, encoding, multiple_targ)
                                encs_targ2 = elmo.encode(model_loaded,
                                                         sents_targ2, stimuli_targ2, encoding, multiple_targ)
                                encs_attr1 = elmo.encode(model_loaded,
                                                         sents_attr1, stimuli_attr1, encoding, multiple_attr)
                                encs_attr2 = elmo.encode(model_loaded,
                                                         sents_attr2, stimuli_attr2, encoding, multiple_attr)
                            elif model == 'bert':
                                encs_targ1 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ1, stimuli_targ1, encoding, multiple_targ)
                                encs_targ2 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ2, stimuli_targ2, encoding, multiple_targ)
                                encs_attr1 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr1, stimuli_attr1, encoding, multiple_attr)
                                encs_attr2 = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr2, stimuli_attr2, encoding, multiple_attr)
                            elif model == 'gpt2':
                                encs_targ1 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ1, stimuli_targ1, encoding, multiple_targ)
                                encs_targ2 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_targ2, stimuli_targ2, encoding, multiple_targ)
                                encs_attr1 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr1, stimuli_attr1, encoding, multiple_attr)
                                encs_attr2 = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents_attr2, stimuli_attr2, encoding, multiple_attr)
                            else:
                                raise ValueError("No encodings computed!")

                            encs = {}
                            encs["targ1"] = {'concept': stimuli['targ1']['concept'], 'encs': encs_targ1}
                            encs["targ2"] = {'concept': stimuli['targ2']['concept'], 'encs': encs_targ2}
                            encs["attr1"] = {'concept': stimuli['attr1']['concept'], 'encs': encs_attr1}
                            encs["attr2"] = {'concept': stimuli['attr2']['concept'], 'encs': encs_attr2}

                            # default parameter: n_samples = 100,000
                            esize, pval = weat.run_test(encs, parametric)
                            results.append(dict(
                                method='SEAT',
                                test=test,
                                model=model,
                                evaluation_measure=measure,
                                context=context,
                                encoding_level=encoding,
                                p_value=pval,
                                effect_size=esize))

                # TODO measure == prob
                elif measure == 'prob':
                    pass

    return results
