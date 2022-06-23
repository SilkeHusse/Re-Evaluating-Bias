import os
import json
import datetime

from methods.LPBS import logprob, generate_sent
from methods.LPBS.encoders import bert, gpt2

TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'data')

def main(models, tests, contexts, evaluations):
    """ Main function of LPBS method """

    # original LPBS method only for simplified and reduced dataset
    tests_allowed_reduced = ['C1_name_word', 'C3_name_word', 'C6_name_word', 'C6_term_word', 'C9_name_word',
                             'C9_name_m_word', 'C9_term_word', 'Occ_name_word', 'Occ_term_word']
    tests_allowed_simpl = ['C1_name_word', 'C3_term_word', 'C6_term_word', 'C9_name_word', 'C9_name_m_word',
                           'Occ_term_word']
    tests_reduced = list(set(tests).intersection(tests_allowed_reduced))
    tests_simpl = list(set(tests).intersection(tests_allowed_simpl))

    results = []

    for model in models:

        if model == 'elmo':
            print(f'Model {model} for method LPBS is not applicable and thus skipped.')
            continue
        elif model == 'bert':
            model_loaded, tokenizer_loaded, subword_tokenizer_loaded = bert.load_model()
        elif model == 'gpt2':
            print(f'Model {model} for method LPBS is not applicable and thus skipped.')
            continue
            #model_loaded, tokenizer_loaded, subword_tokenizer_loaded = gpt2.load_model()
        else:
            raise ValueError("Model %s not found!" % model)

        for test in tests:

            print(f'Computing LPBS for bias test {test}')
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))

            for measure in evaluations:

                if measure == 'cos':
                    pass
                    #for context in contexts:
                    #    if context == 'template':
                    #    elif context == 'reddit':
                    #    else:
                    #        raise ValueError("Context %s not found!" % context)

                elif measure == 'prob':

                    for context in contexts:

                        if context == 'template':

                            # load template sentences
                            template_sents = json.load(open(
                                os.path.join(data_dir, '%s%s' % ('template_double', TEST_EXT)), 'r'))

                            if test in tests_reduced: # case: reduced dataset

                                dataset = json.load(open(os.path.join(
                                    data_dir, 'LPBS/reduced/%s%s' % (test, TEST_EXT)), 'r'))
                                final_template = generate_sent.replace(test, dataset, template_sents, 'reduced')
                                esize, pval = logprob.logprob_cal(model, model_loaded, tokenizer_loaded,
                                                                  subword_tokenizer_loaded, final_template)
                                results.append(dict(
                                    method='LPBS',
                                    test=test,
                                    model=model,
                                    dataset='reduced',
                                    evaluation_metric=measure,
                                    context=context,
                                    encoding_level='',
                                    p_value=pval,
                                    effect_size=esize,
                                    SE='',
                                    SD='',
                                    SD_weighted=''))

                            if test in tests_simpl: # case: simplified dataset

                                dataset = json.load(open(os.path.join(
                                        data_dir, 'LPBS/simplified/%s%s' % (test, TEST_EXT)), 'r'))
                                final_template = generate_sent.replace(test, dataset, template_sents, 'simplified')
                                esize, pval = logprob.logprob_cal(model, model_loaded, tokenizer_loaded,
                                                                  subword_tokenizer_loaded, final_template)
                                results.append(dict(
                                    method='LPBS',
                                    test=test,
                                    dataset='simplified',
                                    model=model,
                                    evaluation_metric=measure,
                                    context=context,
                                    encoding_level='',
                                    p_value=pval,
                                    effect_size=esize,
                                    SE='',
                                    SD='',
                                    SD_weighted=''))

                            # case: full dataset
                            dataset = json.load(open(os.path.join(data_dir, '%s%s' % (test, TEST_EXT)), 'r'))
                            final_template = generate_sent.replace(test, dataset, template_sents)
                            esize, pval = logprob.logprob_cal(model, model_loaded, tokenizer_loaded,
                                                              subword_tokenizer_loaded, final_template)
                            results.append(dict(
                                method='LPBS',
                                test=test,
                                model=model,
                                dataset='full',
                                evaluation_metric=measure,
                                context=context,
                                encoding_level='',
                                p_value=pval,
                                effect_size=esize,
                                SE='',
                                SD='',
                                SD_weighted=''))

                        elif context == 'reddit':
                            print(f'For context {context} no results can be generated at runtime and thus is skipped.')
                            print(f'Please see the results folder directly or execute a respective generate_ebd_* file.')
                            continue

                        else:
                            raise ValueError("Context %s not found!" % context)

    return results
