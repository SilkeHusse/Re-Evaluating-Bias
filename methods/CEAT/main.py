import os
import json
import datetime
import csv

from methods.CEAT import ceat, generate_sent
from methods.CEAT.encoders import elmo, bert, gpt2

TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'data')
result_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'results')

def main(models, tests, encodings, contexts, evaluations):
    """ Main function of CEAT method """

    results = []
    for model in models:

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

            print(f'Computing CEAT for bias test {test}')
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))

            # TODO: indicate if reduced word sets should be used
            reduced_wd_sets = False
            # TODO: indicate if simplified word sets should be used
            simpl_wd_sets = False

            # load stimuli dataset
            if reduced_wd_sets:
                try:
                    dataset = json.load(open(os.path.join(data_dir, 'LPBS/reduced/%s%s' % (test, TEST_EXT)), 'r'))
                    dataset_form = 'reduced'
                except:
                    continue
            elif simpl_wd_sets:
                try:
                    dataset = json.load(open(os.path.join(data_dir, 'LPBS/simplified/%s%s' % (test, TEST_EXT)), 'r'))
                    dataset_form = 'simplified'
                except:
                    continue
            else:
                dataset = json.load(open(os.path.join(data_dir, '%s%s' % (test, TEST_EXT)), 'r'))
                dataset_form = 'full'

            for measure in evaluations:

                if measure == 'cos':

                    for context in contexts:

                        if context == 'template':
                            # load template sentences
                            template_sents = json.load(open(
                                os.path.join(data_dir, '%s%s' % ('template_single', TEST_EXT)), 'r'))

                            stimuli, sents = generate_sent.replace(test, dataset, template_sents, dataset_form)

                            encs = {'targ1': None, 'targ2': None, 'attr1': None, 'attr2': None}

                            if model == 'elmo':
                                encs['targ1'] = elmo.encode(model_loaded, sents['targ1'], stimuli['targ1'], encodings)
                                encs['targ2'] = elmo.encode(model_loaded, sents['targ2'], stimuli['targ2'], encodings)
                                encs['attr1'] = elmo.encode(model_loaded, sents['attr1'], stimuli['attr1'], encodings)
                                encs['attr2'] = elmo.encode(model_loaded, sents['attr2'], stimuli['attr2'], encodings)
                            elif model == 'bert':
                                encs['targ1'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ1'], stimuli['targ1'], encodings)
                                encs['targ2'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ2'], stimuli['targ2'], encodings)
                                encs['attr1'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr1'], stimuli['attr1'], encodings)
                                encs['attr2'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr2'], stimuli['attr2'], encodings)
                            elif model == 'gpt2':
                                encs['targ1'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ1'], stimuli['targ1'], encodings)
                                encs['targ2'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ2'], stimuli['targ2'], encodings)
                                encs['attr1'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr1'], stimuli['attr1'], encodings)
                                encs['attr2'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr2'], stimuli['attr2'], encodings)

                        elif context == 'reddit':
                            print(f'For context {context} no results can be generated at runtime and thus is skipped.')
                            print(f'Please see the results folder directly or execute a respective generate_ebd_* file.')
                            continue

                        else:
                            raise ValueError("Context %s not found!" % context)

                        for encoding in encodings:
                            # default parameter: N = 10,000
                            esize, pval, s_error, s_dev, s_dev_weighted, export_data = ceat.ceat_meta(encs, encoding)
                            results.append(dict(
                                method='CEAT',
                                test=test,
                                model=model,
                                dataset=dataset_form,
                                evaluation_metric=measure,
                                context=context,
                                encoding_level=encoding,
                                p_value=pval,
                                effect_size=esize,
                                SE=s_error,
                                SD=s_dev,
                                SD_weighted=s_dev_weighted))

                            # code snippet to save each effect size and visualize distribution
                            #name_csv = result_dir + '/dists/CEAT_'+str(model)+'_'+str(test)+'_'+str(encoding)+'.csv'
                            #with open(name_csv, 'w', newline='') as csv_file:
                            #    wr = csv.writer(csv_file)
                            #    wr.writerow(("effect_size", "var"))
                            #    wr.writerows(export_data)
                            #csv_file.close()

                elif measure == 'prob':
                    for context in contexts:
                        if context == 'template':
                            print(f'For context {context} and evaluation measure {measure} implementation is currently ongoing and meanwhile skipped.')
                            continue
                        elif context == 'reddit':
                            print(f'For context {context} no results can be generated at runtime and thus is skipped.')
                            print(
                                f'Please see the results folder directly or execute a respective generate_ebd_* file.')
                            continue
                        else:
                            raise ValueError("Context %s not found!" % context)

    return results
