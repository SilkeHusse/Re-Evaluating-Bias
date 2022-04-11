import os
import json

from methods.logprob import logprob
from methods.logprob.encoders import elmo, bert, gpt2

TEST_EXT = '.jsonl'
dirname = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'data')

def main(models, tests, encodings, contexts, evaluations, original):
    """ Main function of logprob method """

    # original logprob method only applicable for certain bias tests
    if original:
        tests_allowed_shrunken = ['C1_name_word', 'C3_name_word', 'C6_name_word', 'C6_term_word', 'C9_name_word', 'C9m_name_word',
                     'C9_term_word', 'Occ_name_word', 'Occ_term_word']
        tests_allowed_minimal = ['C1_name_word', 'C3_term_word', 'C6_term_word', 'C9_name_word', 'C9m_name_word', 'Occ_term_word']
        tests_shrunken = list(set(tests).intersection(tests_allowed_shrunken))
        tests_minimal = list(set(tests).intersection(tests_allowed_minimal))

    results = []
    for model in models:
        for test in tests:

            for measure in evaluations:

                # TODO: cosine measure --> is this equivalent to SEAT?
                if measure == 'cosine':
                    for context in contexts:
                        if context == 'template':
                            pass
                        elif context == 'reddit':
                            pass
                            #print(f'For context {context} no results can be generated at runtime and thus is skipped.')
                            #print(f'Please see the results folder directly or execute a respective generate_ebd_* file.')
                            #break
                        else:
                            raise ValueError("Context %s not found!" % context)

                        for encoding in encodings:
                            pass
                            # default parameter: N = 10,000
                            #esize, pval = ceat.ceat_meta(encs, encoding)
                            #results.append(dict(
                            #    method='CEAT',
                            #    test=test,
                            #    model=model,
                            #    evaluation_measure=measure,
                            #    context=context,
                            #    encoding_level=encoding,
                            #    p_value=pval,
                            #    effect_size=esize))

                elif measure == 'prob':
                    for context in contexts:
                        if context == 'template':

                            # load template sentences dataset
                            template_sents = json.load(open(
                                os.path.join(data_dir, '%s%s' % ('template_double', TEST_EXT)), 'r'))

                            # load respective LM
                            if model == 'elmo':
                                model_loaded = elmo.load_model()
                                tokenizer_loaded, subword_tokenizer_loaded = None, None
                            elif model == 'bert':
                                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = bert.load_model()
                            elif model == 'gpt2':
                                model_loaded, tokenizer_loaded, subword_tokenizer_loaded = gpt2.load_model()
                            else:
                                raise ValueError("Model %s not found!" % model)

                            # load stimuli dataset
                            if original:
                                if test in tests_shrunken:
                                    stimuli = json.load(open(os.path.join(
                                            data_dir, 'stimuli_logprob/shrunken_wd_sets/%s%s' % (test, TEST_EXT)), 'r'))

                                    # for each bias test
                                    # adapt template sents for log prob bias score, save with respective stimuli
                                    final_template = {}
                                    if test == 'C1_name_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('TTT', 'the TTT'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                                #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                                #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'C3_name_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent,
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                                #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                                #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'C6_name_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_hobby']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('AAA', 'the AAA'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'C6_term_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_hobby']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('AAA', 'the AAA').replace('TTT', 'the TTT'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test[:2] == 'C9':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent,
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'Occ_name_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('AAA', 'a AAA'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'Occ_term_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('AAA', 'a AAA').replace('TTT', 'the TTT'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    else:
                                        raise ValueError("Shrunken bias test %s not found!" % test)

                                    esize, pval = logprob.logprob_cal(model_loaded, tokenizer_loaded,
                                                                          subword_tokenizer_loaded,
                                                                          None, final_template, original)
                                    if original:
                                        method_name = 'logprob'
                                    else:
                                        method_name = 'm-logprob'

                                    results.append(dict(
                                        method=method_name,
                                        test=test,
                                        wd_set= 'shrunken',
                                        model=model,
                                        evaluation_measure=measure,
                                        context=context,
                                        p_value=pval,
                                        effect_size=esize))
                                if test in tests_minimal:
                                    stimuli = json.load(open(os.path.join(
                                            data_dir,'stimuli_logprob/minimal_wd_sets/%s%s' % (test, TEST_EXT)),'r'))

                                    # for each bias test
                                    # adapt template sents for log prob bias score, save with respective stimuli
                                    final_template = {}
                                    if test == 'C1_name_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('TTT', 'the TTT'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                                #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                                #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                        for sent in template_sents['templates']['plural_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent,
                                                'stimuli_targ1': stimuli['targ1']['examples_plural'],
                                                'stimuli_targ2': stimuli['targ2']['examples_plural'],
                                                'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                                #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                                #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'C3_term_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_person']:
                                            final_template[idx_template_sent] = {
                                                'template': sent,
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                                #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                                #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                        for sent in template_sents['templates']['plural_person']:
                                            final_template[idx_template_sent] = {
                                                'template': sent,
                                                'stimuli_targ1': stimuli['targ1']['examples_plural'],
                                                'stimuli_targ2': stimuli['targ2']['examples_plural'],
                                                'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                                #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                                #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'C6_term_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_hobby']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('AAA', 'the AAA'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                        for sent in template_sents['templates']['plural_hobby']:
                                            final_template[idx_template_sent] = {
                                                'template': sent,
                                                'stimuli_targ1': stimuli['targ1']['examples_plural'],
                                                'stimuli_targ2': stimuli['targ2']['examples_plural'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test[:2] == 'C9':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_disease']:
                                            final_template[idx_template_sent] = {
                                                'template': sent,
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    elif test == 'Occ_term_word':
                                        idx_template_sent = 0
                                        for sent in template_sents['templates']['singular_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('AAA', 'a AAA'),
                                                'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                                'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                        for sent in template_sents['templates']['plural_basic']:
                                            final_template[idx_template_sent] = {
                                                'template': sent.replace('AAA', 'a AAA'),
                                                'stimuli_targ1': stimuli['targ1']['examples_plural'],
                                                'stimuli_targ2': stimuli['targ2']['examples_plural'],
                                                'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                                'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                            idx_template_sent += 1
                                    else:
                                        raise ValueError("Minimal bias test %s not found!" % test)

                                    esize, pval = logprob.logprob_cal(model_loaded, tokenizer_loaded,
                                                                          subword_tokenizer_loaded,
                                                                          None, final_template, original)
                                    if original:
                                        method_name = 'logprob'
                                    else:
                                        method_name = 'm-logprob'
                                    results.append(dict(
                                        method=method_name,
                                        test=test,
                                        wd_set='minimal',
                                        model=model,
                                        evaluation_measure=measure,
                                        context=context,
                                        p_value=pval,
                                        effect_size=esize))

                            else:
                                stimuli = json.load(open(os.path.join(data_dir, '%s%s' % (test, TEST_EXT)), 'r'))

                                # for each bias test
                                # adapt template sents for log prob bias score, save with respective stimuli
                                final_template = {}
                                if test == 'C1_name_word' or test == 'C3_term_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent.replace('TTT', 'the TTT'),
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                            #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                            #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                    for sent in template_sents['templates']['plural_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent,
                                            'stimuli_targ1': stimuli['targ1']['examples_plural'],
                                            'stimuli_targ2': stimuli['targ2']['examples_plural'],
                                            'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                            #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                            #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test == 'C3_name_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent,
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_adjective'],
                                            #'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_adjective']}
                                            #'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test == 'C6_name_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_hobby']:
                                        final_template[idx_template_sent] = {
                                            'template': sent.replace('AAA', 'the AAA'),
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test == 'C6_term_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_hobby']:
                                        final_template[idx_template_sent] = {
                                            'template': sent.replace('AAA', 'the AAA').replace('TTT', 'the TTT'),
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                    for sent in template_sents['templates']['plural_hobby']:
                                        final_template[idx_template_sent] = {
                                            'template': sent,
                                            'stimuli_targ1': stimuli['targ1']['examples_plural'],
                                            'stimuli_targ2': stimuli['targ2']['examples_plural'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test[:2] == 'C9' or test[:3] == 'Dis':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent,
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test == 'Occ_name_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent.replace('AAA', 'a AAA'),
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test == 'Occ_term_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent.replace('AAA', 'a AAA').replace('TTT', 'the TTT'),
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                    for sent in template_sents['templates']['plural_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent.replace('AAA', 'a AAA'),
                                            'stimuli_targ1': stimuli['targ1']['examples_plural'],
                                            'stimuli_targ2': stimuli['targ2']['examples_plural'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test == 'IBD_name_word' or test == 'EIBD_name_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent,
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                elif test == 'IBD_term_word' or test == 'EIBD_term_word':
                                    idx_template_sent = 0
                                    for sent in template_sents['templates']['singular_basic']:
                                        final_template[idx_template_sent] = {
                                            'template': sent.replace('TTT', 'the TTT'),
                                            'stimuli_targ1': stimuli['targ1']['examples_singular'],
                                            'stimuli_targ2': stimuli['targ2']['examples_singular'],
                                            'stimuli_attr1': stimuli['attr1']['examples_singular'],
                                            'stimuli_attr2': stimuli['attr2']['examples_singular']}
                                        idx_template_sent += 1
                                else:
                                    raise ValueError("Bias test %s not found!" % test)

                                for encoding_level in encodings:
                                    esize, pval = logprob.logprob_cal(model_loaded, tokenizer_loaded,
                                                                      subword_tokenizer_loaded,
                                                                      encoding_level, final_template, original)
                                    if original:
                                        method_name = 'logprob'
                                    else:
                                        method_name = 'm-logprob'

                                    results.append(dict(
                                        method=method_name,
                                        test=test,
                                        model=model,
                                        evaluation_measure=measure,
                                        context=context,
                                        # encoding_level=encoding_level, # TODO ?
                                        p_value=pval,
                                        effect_size=esize))

                        # TODO reddit dataset
                        elif context == 'reddit':
                            # TODO: make sure that sentences contain . at the end --> leave it always out
                            pass

    return results
