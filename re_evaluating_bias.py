""" Main script
args:
    - method (str): s-SEAT, w-SEAT, CEAT, LPBS
    - tests (str): file names of stimuli datasets in data folder
    - models (str): elmo, bert (bbc or bbu), gpt2 (small), opt (small), bloom (small)
    - encoding level (str): word (in case of subword tokenization: -average, -start, -end), sent
    - context (str): template, reddit
    - evaluation measure (str): cos, prob
"""

import os
import random
import time
import argparse
import logging as log
import numpy as np

from csv import DictWriter
from enum import Enum

import methods.SEAT.main as SEAT
import methods.CEAT.main as CEAT
import methods.LPBS.main as LPBS

dirname = os.path.dirname(os.path.realpath(__file__))

class MethodName(Enum):
    SENTSEAT = 's-SEAT'
    WORDSEAT = 'w-SEAT'
    CEAT = 'CEAT'
    LOGPROB = 'LPBS'
class ModelName(Enum):
    ELMO = 'elmo'
    BERT = 'bert'
    GPT2 = 'gpt2'
    OPT = 'opt'
    BLOOM = 'bloom'
class EncodingName(Enum):
    WORDAVG = 'word-average'
    WORDSTART = 'word-start'
    WORDEND = 'word-end'
    SENT = 'sent'
class ContextName(Enum):
    TEMPLATE = 'template'
    REDDIT = 'reddit'
class EvaluationName(Enum):
    COS = 'cos'
    PROB = 'prob'

TEST_EXT = '.jsonl'
METHOD_NAMES = [m.value for m in MethodName]
MODEL_NAMES = [m.value for m in ModelName]
ENCODING_NAMES = [m.value for m in EncodingName]
CONTEXT_NAMES = [m.value for m in ContextName]
EVALUATION_NAMES = [m.value for m in EvaluationName]

def handle_arguments(arguments):
    """ Helper function for handling argument parsing"""

    parser = argparse.ArgumentParser(
        description='Run particular bias tests on specified language models for different bias detection methods.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', '-m', type=str,
                        help='Bias detection methods to execute (comma-separated list; options: {}). '
                             'Default: all methods.'.format(','.join(METHOD_NAMES)))
    parser.add_argument('--test', '-t', type=str,
                        help='Bias tests to run (comma-separated list; test files should be in `data` and '
                             'have corresponding names, with extension {}). Default: all tests.'.format(TEST_EXT))
    parser.add_argument('--model', '-l', type=str,
                        help='Language models to evaluate (comma-separated list; options: {}). '
                             'Default: all models.'.format(','.join(MODEL_NAMES)))
    parser.add_argument('--encoding', '-e', type=str,
                        help='Encoding levels to execute (comma-separated list; options: {}). '
                             'Default: word-average.'.format(','.join(ENCODING_NAMES)))
    parser.add_argument('--context', '-c', type=str,
                        help='Contexts to evaluate (comma-separated list; options: {}). '
                             'Default: template.'.format(','.join(CONTEXT_NAMES)))
    parser.add_argument('--evaluation', '-b', type=str,
                        help='Evaluation metrics to run (comma-separated list; options: {}). '
                             'Default: cos.'.format(','.join(EVALUATION_NAMES)))
    parser.add_argument('--log_file', '-f', type=str,
                        help='File to log to')
    parser.add_argument('--parametric', action='store_true',
                        help='Use parametric test (normal assumption). Default: False.')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use CPU to encode sentences. Default: False.')
    return parser.parse_args(arguments)

def check_allowance(arg_str, allowed_set, item_type):
    """ Function to check allowance
    args:
        - arg_str (str): comma separated str of items
        - allowed_set (list): contains allowed items
        - item_type (str): for message purposes
    return list containing items
    """
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError('Unknown %s: %s!' % (item_type, item))
    return items

def main(arguments):
    """ Main function to parse args and run tests for defined specs """
    log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
    args = handle_arguments(arguments)
    if args.log_file:
        log.getLogger().addHandler(log.FileHandler(args.log_file))

    # set seeds for reproducibility
    random.seed(1111)
    np.random.seed(1111)

    methods = check_allowance(args.method, METHOD_NAMES, 'method') if args.method is not None else METHOD_NAMES
    log.info('Methods selected:')
    for method in methods:
        log.info('\t{}'.format(method))

    models = check_allowance(args.model, MODEL_NAMES, 'model') if args.model is not None else MODEL_NAMES
    log.info('Models selected:')
    for model in models:
        log.info('\t{}'.format(model))

    all_tests = sorted([entry[:-len(TEST_EXT)]
                        for entry in os.listdir('data')
                        if not entry.startswith('.') and entry.endswith('word' + TEST_EXT)])
    tests = check_allowance(args.test, all_tests, 'test') if args.test is not None else all_tests
    log.info('Tests selected:')
    for test in tests:
        log.info('\t{}'.format(test))

    encodings = check_allowance(args.encoding, ENCODING_NAMES, 'encoding') if args.encoding is not None else ['word-average']
    log.info('Encoding levels selected:')
    for encoding in encodings:
        log.info('\t{}'.format(encoding))

    contexts = check_allowance(args.context, CONTEXT_NAMES, 'context') if args.context is not None else ['template']
    log.info('Contexts selected:')
    for context in contexts:
        log.info('\t{}'.format(context))

    evaluations = check_allowance(args.evaluation, EVALUATION_NAMES, 'evaluation') if args.evaluation is not None else ['cosine']
    log.info('Evaluation metrics selected:')
    for evaluation in evaluations:
        log.info('\t{}'.format(evaluation))

    results = []
    results_method = []
    for method_name in methods:

        if method_name == MethodName.SENTSEAT.value:
            if any('word' in encoding_level for encoding_level in encodings):
                log.info('Note: word encoding level for method s-SEAT is equivalent to method w-SEAT.')
                log.info('Note: probability as evaluation metric for method s-SEAT is not applicable and thus skipped.')
            results_method = SEAT.main(models, tests, encodings, contexts, evaluations, args.parametric)
        elif method_name == MethodName.WORDSEAT.value:
            if any('sent' in encoding_level for encoding_level in encodings):
                log.info('Note: sentence encoding level for method w-SEAT is equivalent to method s-SEAT.')
                log.info('Note: probability as evaluation metric for method w-SEAT is not applicable and thus skipped.')
            results_method = SEAT.main(models, tests, encodings, contexts, evaluations, args.parametric)
        elif method_name == MethodName.CEAT.value:
            results_method = CEAT.main(models, tests, encodings, contexts, evaluations)
        elif method_name == MethodName.LOGPROB.value:
            log.info('Note: encoding level for method LPBS is not applicable and thus skipped.')
            log.info('Note: cosine similarity as evaluation metric for method LPBS is not applicable and thus skipped.')
            results_method = LPBS.main(models, tests, contexts, evaluations)
        results = results + results_method

    # save results and specs of code run (time, date)
    results_path = dirname + '/results/' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
    log.info('Writing results to {}'.format(results_path))
    with open(results_path, 'w') as f:
        writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

# uncomment for console usage
#import sys
#if __name__ == '__main__':
#    main(sys.argv[1:])

main(['-mw-SEAT',
      #'-tC1_name_word,C3_name_word,C3_term_word,C6_name_word,C6_term_word,C9_name_word,C9_name_m_word,C9_term_word,Occ_name_word,Occ_term_word,Dis_term_word,Dis_term_m_word,I1_name_word,I1_term_word,I2_name_word,I2_term_word',
      '-tC1_name_word,C6_name_word',
      '-lbloom',
      '-eword-average',
      '-ctemplate',
      '-bcos'])