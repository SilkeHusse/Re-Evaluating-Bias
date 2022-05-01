import os
import json
import datetime

from methods.CEAT import ceat, generate_sent
from methods.CEAT.encoders import elmo, bert, gpt2

TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'data')

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
            # TODO: indicate if minimal word sets should be used
            simpl_wd_sets = False

            # load stimuli dataset
            if reduced_wd_sets:
                try:
                    dataset = json.load(open(os.path.join(data_dir, 'LPBS/reduced/%s%s' % (test, TEST_EXT)), 'r'))
                except:
                    continue
            elif simpl_wd_sets:
                try:
                    dataset = json.load(open(os.path.join(data_dir, 'LPBS/simplified/%s%s' % (test, TEST_EXT)), 'r'))
                except:
                    continue
            else:
                dataset = json.load(open(os.path.join(data_dir, '%s%s' % (test, TEST_EXT)), 'r'))

            for measure in evaluations:

                if measure == 'cos':

                    for context in contexts:

                        if context == 'template':
                            # load template sentences
                            template_sents = json.load(open(
                                os.path.join(data_dir, '%s%s' % ('template_single', TEST_EXT)), 'r'))

                            if reduced_wd_sets:
                                stimuli, sents = generate_sent.replace(test, dataset, template_sents, 'reduced')
                            elif simpl_wd_sets:
                                stimuli, sents = generate_sent.replace(test, dataset, template_sents, 'simplified')
                            else:
                                stimuli, sents = generate_sent.replace(test, dataset, template_sents)

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
                            break

                        else:
                            raise ValueError("Context %s not found!" % context)

                        for encoding in encodings:
                            # default parameter: N = 10,000
                            esize, pval = ceat.ceat_meta(encs, encoding)
                            results.append(dict(
                                method='CEAT',
                                test=test,
                                model=model,
                                evaluation_metric=measure,
                                context=context,
                                encoding_level=encoding,
                                p_value=pval,
                                effect_size=esize))

                elif measure == 'prob':
                    pass # TODO

    return results
