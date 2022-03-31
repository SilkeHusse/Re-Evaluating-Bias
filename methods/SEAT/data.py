""" Helper function related to loading data """
import json

WEAT_SETS = ["targ1", "targ2", "attr1", "attr2"]
CONCEPT = "concept"

def load_json(sent_file):
    """ Load data from jsonl file """
    return json.load(open(sent_file, 'r'))
