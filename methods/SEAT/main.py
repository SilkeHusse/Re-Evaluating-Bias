import os
import random
import json
import datetime

from methods.SEAT import weat, generate_sent
# TODO: set respective LMs
#from methods.SEAT.encoders import bert, elmo, gpt2
from methods.SEAT.encoders import opt, bloom

random.seed(1111)

TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'data')
result_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'results')

def main(models, tests, encodings, contexts, evaluations, parametric):
    """ Main function of SEAT method """

    results = []

    for model in models:

        if contexts != ['reddit']: # only load model if required
            if model == 'elmo':
                model_loaded = elmo.load_model()
                tokenizer_loaded, subword_tokenizer_loaded = None, None
            elif model == 'bert':
                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = bert.load_model()
            elif model == 'gpt2':
                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = gpt2.load_model()
            elif model == 'opt':
                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = opt.load_model()
            elif model == 'bloom':
                model_loaded, tokenizer_loaded = bloom.load_model()
            else:
                raise ValueError("Model %s not found!" % model)
        else:
            model_loaded, tokenizer_loaded, subword_tokenizer_loaded = None, None, None

        for test in tests:

            runtimes = []

            print(f'Computing SEAT for bias test {test}')
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
                            template_sents = json.load(open(os.path.join(data_dir, '%s%s' %
                                                                         ('template_single', TEST_EXT)), 'r'))

                            # for each bias test
                            # - extract stimuli from file
                            # - create sents by replacing target and attribute words in template sentences
                            stimuli, sents, multiple_targ, multiple_attr = generate_sent.replace(test, dataset, template_sents, dataset_form)

                            # target sets have to be of equal size
                            if not len(sents['targ1']) == len(sents['targ2']):
                                min_n = min([len(sents['targ1']), len(sents['targ2'])])
                                # randomly sample min number of sents for both word sets
                                if not len(sents['targ1']) == min_n:
                                    sents['targ1'] = random.sample(sents['targ1'], min_n)
                                else:
                                    sents['targ2'] = random.sample(sents['targ2'], min_n)

                        elif context == 'reddit':
                            print(f'For context {context} no results can be generated at runtime and thus is skipped.')
                            print(f'Please see the results folder directly or execute a respective generate_ebd_* file.')
                            continue

                        else:
                            raise ValueError("Context %s not found!" % context)

                        for encoding in encodings:

                            encs = {'targ1': None, 'targ2': None, 'attr1': None, 'attr2': None}

                            if model == 'elmo':
                                encs['targ1'] = elmo.encode(model_loaded,
                                                         sents['targ1'], stimuli['targ1'], encoding, multiple_targ)
                                encs['targ2'] = elmo.encode(model_loaded,
                                                         sents['targ2'], stimuli['targ2'], encoding, multiple_targ)
                                encs['attr1'] = elmo.encode(model_loaded,
                                                         sents['attr1'], stimuli['attr1'], encoding, multiple_attr)
                                encs['attr2'] = elmo.encode(model_loaded,
                                                         sents['attr2'], stimuli['attr2'], encoding, multiple_attr)
                            elif model == 'bert':
                                encs['targ1'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ1'], stimuli['targ1'], encoding, multiple_targ)
                                encs['targ2'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ2'], stimuli['targ2'], encoding, multiple_targ)
                                encs['attr1'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr1'], stimuli['attr1'], encoding, multiple_attr)
                                encs['attr2'] = bert.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr2'], stimuli['attr2'], encoding, multiple_attr)
                            elif model == 'gpt2':
                                encs['targ1'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ1'], stimuli['targ1'], encoding, multiple_targ)
                                encs['targ2'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['targ2'], stimuli['targ2'], encoding, multiple_targ)
                                encs['attr1'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr1'], stimuli['attr1'], encoding, multiple_attr)
                                encs['attr2'] = gpt2.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                         sents['attr2'], stimuli['attr2'], encoding, multiple_attr)
                            elif model == 'opt':
                                encs['targ1'] = opt.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                            sents['targ1'], stimuli['targ1'], encoding, multiple_targ)
                                encs['targ2'] = opt.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                            sents['targ2'], stimuli['targ2'], encoding, multiple_targ)
                                encs['attr1'] = opt.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                            sents['attr1'], stimuli['attr1'], encoding, multiple_attr)
                                encs['attr2'] = opt.encode(model_loaded, tokenizer_loaded, subword_tokenizer_loaded,
                                                            sents['attr2'], stimuli['attr2'], encoding, multiple_attr)
                            elif model == 'bloom':
                                encs['targ1'] = bloom.encode(model_loaded, tokenizer_loaded,
                                                            sents['targ1'], stimuli['targ1'], encoding, multiple_targ)
                                encs['targ2'] = bloom.encode(model_loaded, tokenizer_loaded,
                                                            sents['targ2'], stimuli['targ2'], encoding, multiple_targ)
                                encs['attr1'] = bloom.encode(model_loaded, tokenizer_loaded,
                                                            sents['attr1'], stimuli['attr1'], encoding, multiple_attr)
                                encs['attr2'] = bloom.encode(model_loaded, tokenizer_loaded,
                                                            sents['attr2'], stimuli['attr2'], encoding, multiple_attr)
                            else:
                                raise ValueError("No encodings computed!")

                            #for iteration in range(10):

                                # datetime object for runtime
                                #start = datetime.datetime.now()

                            # default parameter: n_samples = 100,000
                            esize, pval = weat.run_test(encs, parametric)
                            results.append(dict(
                                method='SEAT',
                                test=test,
                                model=model,
                                dataset=dataset_form,
                                evaluation_metric=measure,
                                context=context,
                                encoding_level=encoding,
                                p_value=pval,
                                effect_size=esize,
                                SE='',
                                SD='',
                                SD_weighted=''))

                                # datetime object for runtime
                                #end = datetime.datetime.now()
                                #delta_time = end-start

                                #runtimes.append([start, end, delta_time])

                elif measure == 'prob':
                    pass

            # code snippet to save runtimes
            #specs = 'SEAT' + '_' + str(test[:-5]) + '_' + str(model) + '.txt'
            #with open(os.path.join(result_dir, 'runtime/%s' % specs), 'w') as file:
            #    for item in runtimes:
            #        file.write(item[0].strftime("%d-%m-%Y (%H:%M:%S.%f)" + '\n'))
            #        file.write(item[1].strftime("%d-%m-%Y (%H:%M:%S.%f)" + '\n'))
            #        file.write(str(item[2]) + '\n')

    return results
