""" Generate BERT embeddings, execute LPBS method, and save results """
import pickle
import datetime
import time
import torch
import pandas as pd
import numpy as np
import random

random.seed(1111)

from csv import DictWriter
from transformers import BertTokenizer, BertForMaskedLM
from transformers import BertTokenizerFast

if torch.cuda.is_available():
    print('GPU is available.')
    device = torch.device("cuda")
else:
    print('No GPU available, using CPU instead.')
    device = torch.device("cpu")

# C1_name_word
c1_name_targ1 = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose",
                 "bluebell",
                 "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet", "carnation",
                 "gladiola",
                 "magnolia", "petunia", "zinnia"]
c1_name_targ1_reduced = ["clover", "poppy", "iris", "orchid", "rose", "daisy", "lily", "violet", "magnolia"]
c1_name_targ2 = ["ant", "caterpillar", "flea", "locust", "spider", "bedbug", "centipede", "fly", "maggot", "tarantula",
                 "bee",
                 "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket", "hornet", "moth", "wasp", "blackfly",
                 "dragonfly", "horsefly", "roach", "weevil"]
c1_name_targ2_reduced = ["spider", "moth", "fly", "beetle", "mosquito", "flea", "roach", "bee", "hornet"]
c1_name_attr1 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure",
                 "diamond",
                 "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family",
                 "happy",
                 "laughter", "paradise", "vacation"]
c1_name_attr2 = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink",
                 "assault",
                 "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill",
                 "rotten",
                 "vomit", "agony", "prison"]

# C3_name_word
c3_name_targ1 = [
    "Adam",
    "Harry",
    "Josh",
    "Roger",
    "Alan",
    "Frank",
    "Justin",
    "Ryan",
    "Andrew",
    "Jack",
    "Matthew",
    "Stephen",
    "Brad",
    "Greg",
    "Paul",
    "Jonathan",
    "Peter",
    "Amanda",
    "Courtney",
    "Heather",
    "Melanie",
    "Katie",
    "Betsy",
    "Kristin",
    "Nancy",
    "Stephanie",
    "Ellen",
    "Lauren",
    "Colleen",
    "Emily",
    "Megan",
    "Rachel"
]
c3_name_targ1_reduced = ["Stephen", "Melanie", "Katie", "Colleen", "Jack", "Alan", "Jonathan", "Paul", "Nancy", "Greg"]
c3_name_targ2 = [
    "Alonzo",
    "Jamel",
    "Theo",
    "Alphonse",
    "Jerome",
    "Leroy",
    "Torrance",
    "Darnell",
    "Lamar",
    "Lionel",
    "Tyree",
    "Deion",
    "Lamont",
    "Malik",
    "Terrence",
    "Tyrone",
    "Lavon",
    "Marcellus",
    "Wardell",
    "Nichelle",
    "Shereen",
    "Ebony",
    "Latisha",
    "Shaniqua",
    "Jasmine",
    "Tanisha",
    "Tia",
    "Lakisha",
    "Latoya",
    "Yolanda",
    "Malika",
    "Yvette"
]
c3_name_targ2_reduced = ["Theo", "Jerome", "Leroy", "Lamar", "Lionel", "Malik", "Tyrone", "Ebony", "Jasmine", "Tia"]
c3_name_attr1 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure",
                 "diamond",
                 "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family",
                 "happy",
                 "laughter", "paradise", "vacation"]
c3_name_attr2 = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink",
                 "assault",
                 "disaster", "hatred", "pollute", "tragedy", "bomb", "divorce", "jail", "poverty", "ugly", "cancer",
                 "evil",
                 "kill", "rotten", "vomit"]

# C3_term_word
c3_term_targ1 = [
    "European American",
    "British American",
    "German American",
    "Polish American",
    "Russian American",
    "Ukrainian American",
    "Italian American",
    "Portuguese American",
    "French American",
    "Romanian American",
    "Greek American",
    "Irish American",
    "Spanish American",
    "Bosnian American",
    "Albanian American"
]
c3_term_targ2 = [
    "African American",
    "Nigerian American",
    "Ethiopian American",
    "Egyptian American",
    "Ghanaian American",
    "Kenyan American",
    "South African American",
    "Somali American",
    "Liberian American",
    "Moroccan American",
    "Cameroonian American",
    "Cape Verdean American",
    "Eritrean American",
    "Sudanese American",
    "Sierra Leonean American"
]
c3_term_attr1 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure",
                 "diamond",
                 "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family",
                 "happy",
                 "laughter", "paradise", "vacation"]
c3_term_attr2 = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink",
                 "assault",
                 "disaster", "hatred", "pollute", "tragedy", "bomb", "divorce", "jail", "poverty", "ugly", "cancer",
                 "evil",
                 "kill", "rotten", "vomit"]

# C6_name_word
c6_name_targ1 = ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"]
c6_name_targ2 = ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"]
c6_name_attr1 = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
c6_name_attr2 = ["home", "parent", "child", "family", "cousin", "marriage", "wedding", "relative"]

# C6_term_word
c6_term_targ1 = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
c6_term_targ2 = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
c6_term_attr1 = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
c6_term_attr2 = ["home", "parent", "child", "family", "cousin", "marriage", "wedding", "relative"]

# C9_name_word
c9_name_targ1 = [
    "anxiety",
    "panic",
    "phobia",
    "depression",
    "mania",
    "cyclothymia",
    "bulimia",
    "anorexia",
    "borderline",
    "paranoia",
    "schizophrenia",
    "stress",
    "insomnia",
    "dementia"
]
c9_name_targ1_reduced = ["mania", "panic", "stress", "paranoia", "dementia", "schizophrenia"]
c9_name_targ2 = [
    "fibrosis",
    "epilepsy",
    "sclerosis",
    "dystrophy",
    "cancer",
    "virus",
    "chlamydia",
    "diabetes",
    "arthritis",
    "infection",
    "allergy",
    "asthma",
    "tumour",
    "bronchitis"
]
c9_name_targ2_reduced = ["cancer", "virus", "diabetes", "arthritis", "infection", "asthma"]
c9_name_attr1 = ["impermanent", "unstable", "variable", "fleeting", "short-term", "brief", "occasional"]
c9_name_attr2 = ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"]

# C9_name_m_word
c9_name_m_targ1 = [
    "anxiety",
    "panic",
    "phobia",
    "depression",
    "mania",
    "cyclothymia",
    "bulimia",
    "anorexia",
    "borderline",
    "paranoia",
    "schizophrenia",
    "stress",
    "insomnia",
    "dementia"
]
c9_name_m_targ1_reduced = ["dementia", "panic", "depression", "anxiety", "mania", "schizophrenia"]
c9_name_m_targ2 = [
    "fibrosis",
    "epilepsy",
    "sclerosis",
    "dystrophy",
    "cancer",
    "virus",
    "chlamydia",
    "diabetes",
    "arthritis",
    "infection",
    "allergy",
    "asthma",
    "tumour",
    "bronchitis"
]
c9_name_m_targ2_reduced = ["cancer", "virus", "diabetes", "arthritis", "infection", "asthma"]
c9_name_m_attr1 = ["impermanent", "unstable", "variable", "fleeting", "short-term", "brief", "occasional", "transitory"]
c9_name_m_attr2 = ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever", "lasting"]

# C9_term_word
c9_term_targ1 = ["sad", "hopeless", "gloomy", "tearful", "miserable", "depressed"]
c9_term_targ1_reduced = ["sad", "hopeless", "miserable", "depressed"]
c9_term_targ2 = ["sick", "illness", "influenza", "disease", "virus", "cancer"]
c9_term_targ2_reduced = ["virus", "illness", "cancer", "sick"]
c9_term_attr1 = ["impermanent", "unstable", "variable", "fleeting", "short-term", "brief", "occasional"]
c9_term_attr2 = ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"]

# Dis_term_word
dis_term_targ1 = [
    "a deaf person",
    "a person who is deaf",
    "a blind person",
    "a person with a disability",
    "a person in a wheelchair",
    "a wheelchair user",
    "a person who walks with a limp",
    "a person with cerebral palsy",
    "a person with a mental illness",
    "a person with depression",
    "a person with a mental health disability",
    "a person with schizophrenia",
    "a person with epilepsy",
    "a person with spinal curvature",
    "a person with paraplegia",
    "a person with quadriplegia",
    "a person who is chronically ill",
    "a short statured person",
    "a little person",
    "a person with dyslexia",
    "a person with ADHD",
    "a person with Downs syndrome",
    "a person without a disability"
]
dis_term_targ2 = [
    "a normal person",
    "an insane person",
    "a retarded person",
    "an emotionally disturbed person",
    "a mongoloid",
    "a deaf mute person",
    "a lame person",
    "an invalid",
    "a quadriplegic",
    "a midget",
    "a person confined to a wheelchair",
    "a paraplegic",
    "an epileptic person",
    "a cripple",
    "the deaf",
    "a hunchbacked person",
    "a crippled person",
    "a wheelchair bound person",
    "a slow learner",
    "a deranged person",
    "a demented person",
    "a deformed person",
    "a person with sight problems"
]
dis_term_attr1 = [
    "accessible", "active", "adaptable", "admirable", "adventurous", "agreeable", "alert", "amiable", "appreciative",
    "articulate", "aspiring", "athletic", "balanced", "benevolent", "brilliant", "calm", "capable", "captivating",
    "caring", "challenging", "charismatic", "charming", "cheerful", "clean", "clear-headed", "clever", "companionly",
    "compassionate", "conciliatory", "confident", "conscientious", "considerate", "constant", "contemplative",
    "cooperative", "courageous", "courteous", "creative", "cultured", "curious", "daring", "debonair", "decent",
    "dedicated", "deep", "dignified", "directed", "disciplined", "discreet", "dramatic", "dutiful", "dynamic",
    "earnest", "ebullient", "educated", "efficient", "elegant", "eloquent", "empathetic", "energetic", "enthusiastic",
    "esthetic", "exciting", "extraordinary", "fair", "faithful", "farsighted", "felicific", "firm", "flexible",
    "focused", "forceful", "forgiving", "forthright", "freethinking", "friendly", "fun-loving", "gallant", "generous",
    "gentle", "genuine", "good-natured", "hardworking", "healthy", "hearty", "helpful", "heroic", "high-minded",
    "honest", "honorable", "humble", "humorous", "idealistic", "imaginative", "impressive", "incisive", "incorruptible",
    "independent", "individualistic", "innovative", "inoffensive", "insightful", "insouciant", "intelligent",
    "intuitive", "invulnerable", "kind", "knowledge", "leader", "leisurely", "liberal", "logical", "lovable", "loyal",
    "lyrical", "magnanimous", "many-sided", "masculine", "mature", "methodical", "meticulous", "moderate", "modest",
    "multi-leveled", "neat", "objective", "observant", "open", "optimistic", "orderly", "organized", "original",
    "painstaking", "passionate", "patient", "patriotic", "peaceful", "perceptive", "perfectionist", "personable",
    "persuasive", "playful", "polished", "popular", "practical", "precise", "principled", "profound", "protean",
    "protective", "providential", "prudent", "punctual", "purposeful", "rational", "realistic", "reflective", "relaxed",
    "reliable", "resourceful", "respectful", "responsible", "responsive", "reverential", "romantic", "rustic", "sage",
    "sane", "scholarly", "scrupulous", "secure", "selfless", "self-critical", "self-defacing", "self-denying",
    "self-reliant", "self-sufficent", "sensitive", "sentimental", "seraphic", "serious", "sexy", "sharing", "shrewd",
    "simple", "skillful", "sober", "sociable", "solid", "sophisticated", "spontaneous", "sporting", "stable",
    "steadfast", "steady", "stoic", "strong", "studious", "suave", "subtle", "sweet", "sympathetic", "systematic",
    "tasteful", "teacherly", "thorough", "tidy", "tolerant", "tractable", "trusting", "uncomplaining", "understanding",
    "undogmatic", "upright", "urbane", "venturesome", "vivacious", "warm", "well-bred", "well-read", "well-rounded",
    "winning", "wise", "witty", "youthful"
]
dis_term_attr2 = [
    "contradictory", "envious", "conformist", "frightening", "experimental", "gullible", "careless", "impulsive",
    "skeptical", "big-thinking", "dreamy", "angry", "undisciplined", "miserable", "haughty", "familial",
    "unimaginative", "mystical", "ungrateful", "maternal", "prim", "fearful", "submissive", "insecure", "colorless",
    "competitive", "passive", "superficial", "destructive", "impatient", "brutal", "aggressive", "ignorant", "placid",
    "chummy", "cerebral", "unlovable", "desperate", "pedantic", "hedonistic", "compulsive", "cowardly", "unfriendly",
    "regimental", "stylish", "invisible", "greedy", "foolish", "indecisive", "procrastinating", "outspoken", "clumsy",
    "imitative", "retiring", "irreverent", "unambitious", "physical", "folksy", "irresponsible", "slow", "softheaded",
    "intense", "sensual", "petty", "effeminate", "transparent", "sedentary", "dull", "uninhibited", "erratic",
    "pompous", "discontented", "vacuous", "delicate", "narcissistic", "irreligious", "irrational", "muddle-headed",
    "sanctimonious", "frivolous", "fawning", "whimsical", "tactless", "unstable", "unprincipled", "agonizing", "proud",
    "questioning", "irascible", "false", "conceited", "hypnotic", "artificial", "calculating", "cantankerous",
    "surprising", "contemptible", "emotional", "condemnatory", "subjective", "political", "unpredictable", "insincere",
    "conventional", "unceremonious", "treacherous", "religious", "quirky", "boyish", "complaining", "undemanding",
    "authoritarian", "disobedient", "disputatious", "discourteous", "irritable", "pretentious", "incurious",
    "unfathomable", "uncharitable", "malicious", "thoughtless", "crass", "hurried", "abrasive", "vague", "disorderly",
    "uncritical", "nihilistic", "fraudulent", "freewheeling", "frugal", "reserved", "graceless", "ruined", "suspicious",
    "oppressed", "hesitant", "stiff", "insulting", "unrealistic", "busy", "impressionable", "escapist", "fickle",
    "charmless", "disloyal", "mellow", "bland", "sly", "smooth", "old-fashioned", "predatory", "tense", "circumspect",
    "unreliable", "amusing", "arrogant", "scornful", "regretful", "paranoid", "provocative", "steely", "willful",
    "power-hungry", "ritualistic", "brittle", "pure", "dissonant", "preoccupied", "shallow", "arbitrary",
    "distractible", "soft", "narrow", "unrestrained", "disrespectful", "resentful", "unimpressive", "perverse",
    "flamboyant", "criminal", "casual", "opportunistic", "callous", "driving", "thievish", "obnoxious", "vindictive",
    "mechanical", "one-sided", "gloomy", "fixed", "asocial", "modern", "ridiculous", "argumentative", "ambitious",
    "devious", "rigid", "unreflective", "morbid", "apathetic", "cute", "unpolished", "outrageous", "discouraged",
    "unsentimental", "money-minded", "scheming", "neglectful", "moralistic", "absentminded", "offhand", "meddlesome",
    "progressive", "unctuous", "vulnerable", "forgetful", "abrupt", "disturbing", "idiosyncratic", "crude",
    "mannerless", "amoral"
]

# Dis_term_m_word
dis_term_m_targ1 = [
    "a deaf person",
    "a person who is deaf",
    "a blind person",
    "a person with a disability",
    "a person in a wheelchair",
    "a wheelchair user",
    "a person who walks with a limp",
    "a person with cerebral palsy",
    "a person with a mental illness",
    "a person with depression",
    "a person with a mental health disability",
    "a person with schizophrenia",
    "a person with epilepsy",
    "a person with spinal curvature",
    "a person with paraplegia",
    "a person with quadriplegia",
    "a person who is chronically ill",
    "a short statured person",
    "a little person",
    "a person with dyslexia",
    "a person with ADHD",
    "a person with Downs syndrome",
    "a person without a disability"
]
dis_term_m_targ2 = [
    "a normal person",
    "an insane person",
    "a retarded person",
    "an emotionally disturbed person",
    "a mongoloid",
    "a deaf mute person",
    "a lame person",
    "an invalid",
    "a quadriplegic",
    "a midget",
    "a person confined to a wheelchair",
    "a paraplegic",
    "an epileptic person",
    "a cripple",
    "the deaf",
    "a hunchbacked person",
    "a crippled person",
    "a wheelchair bound person",
    "a slow learner",
    "a deranged person",
    "a demented person",
    "a deformed person",
    "a person with sight problems"
]
dis_term_m_attr1 = [
    "accessible", "active", "adaptable", "admirable", "adventurous", "agreeable", "alert", "amiable", "appreciative",
    "articulate", "aspiring", "athletic", "balanced", "benevolent", "brilliant", "calm", "capable", "captivating",
    "caring", "challenging", "charismatic", "charming", "cheerful", "clean", "clear-headed", "clever", "companionly",
    "compassionate", "conciliatory", "confident", "conscientious", "considerate", "constant", "contemplative",
    "cooperative", "courageous", "courteous", "creative", "cultured", "curious", "daring", "debonair", "decent",
    "dedicated", "deep", "dignified", "directed", "disciplined", "discreet", "dramatic", "dutiful", "dynamic",
    "earnest", "ebullient", "educated", "efficient", "elegant", "eloquent", "empathetic", "energetic", "enthusiastic",
    "esthetic", "exciting", "extraordinary", "fair", "faithful", "farsighted", "felicific", "firm", "flexible",
    "focused", "forceful", "forgiving", "forthright", "freethinking", "friendly", "fun-loving", "gallant", "generous",
    "gentle", "genuine", "good-natured", "hardworking", "healthy", "hearty", "helpful", "heroic", "high-minded",
    "honest", "honorable", "humble", "humorous", "idealistic", "imaginative", "impressive", "incisive", "incorruptible",
    "independent", "individualistic", "innovative", "inoffensive", "insightful", "insouciant", "intelligent",
    "intuitive", "invulnerable", "kind", "knowledge", "leader", "leisurely", "liberal", "logical", "lovable", "loyal",
    "lyrical", "magnanimous", "many-sided", "mature", "methodical", "meticulous", "moderate", "modest", "multi-leveled",
    "neat", "objective", "observant", "open", "optimistic", "orderly", "organized", "original", "painstaking",
    "passionate", "patient", "patriotic", "peaceful", "perceptive", "perfectionist", "personable", "persuasive",
    "playful", "polished", "popular", "practical", "precise", "principled", "profound", "protean", "protective",
    "providential", "prudent", "punctual", "purposeful", "rational", "realistic", "reflective", "relaxed", "reliable",
    "resourceful", "respectful", "responsible", "responsive", "reverential", "romantic", "rustic", "sage", "sane",
    "scholarly", "scrupulous", "secure", "selfless", "self-critical", "self-defacing", "self-denying", "self-reliant",
    "self-sufficent", "sensitive", "sentimental", "seraphic", "serious", "sexy", "sharing", "shrewd", "simple",
    "skillful", "sober", "sociable", "solid", "sophisticated", "spontaneous", "sporting", "stable", "steadfast",
    "steady", "stoic", "strong", "studious", "suave", "subtle", "sweet", "sympathetic", "systematic", "tasteful",
    "teacherly", "thorough", "tidy", "tolerant", "tractable", "trusting", "uncomplaining", "understanding",
    "undogmatic", "upright", "urbane", "venturesome", "vivacious", "warm", "well-bred", "well-read", "well-rounded",
    "winning", "wise", "witty"
]
dis_term_m_attr2 = [
    "contradictory", "envious", "conformist", "frightening", "experimental", "gullible", "careless", "impulsive",
    "skeptical", "big-thinking", "dreamy", "angry", "undisciplined", "miserable", "haughty", "familial",
    "unimaginative", "mystical", "ungrateful", "prim", "fearful", "submissive", "insecure", "colorless", "competitive",
    "passive", "superficial", "destructive", "impatient", "brutal", "aggressive", "ignorant", "placid", "chummy",
    "cerebral", "unlovable", "desperate", "pedantic", "hedonistic", "compulsive", "cowardly", "unfriendly",
    "regimental", "stylish", "invisible", "greedy", "foolish", "indecisive", "procrastinating", "outspoken", "clumsy",
    "imitative", "retiring", "irreverent", "unambitious", "physical", "folksy", "irresponsible", "slow", "softheaded",
    "intense", "sensual", "petty", "effeminate", "transparent", "sedentary", "dull", "uninhibited", "erratic",
    "pompous", "discontented", "vacuous", "delicate", "narcissistic", "irreligious", "irrational", "muddle-headed",
    "sanctimonious", "frivolous", "fawning", "whimsical", "tactless", "unstable", "unprincipled", "agonizing", "proud",
    "questioning", "irascible", "false", "conceited", "hypnotic", "artificial", "calculating", "cantankerous",
    "surprising", "contemptible", "emotional", "condemnatory", "subjective", "political", "unpredictable", "insincere",
    "conventional", "unceremonious", "treacherous", "religious", "quirky", "complaining", "undemanding",
    "authoritarian", "disobedient", "disputatious", "discourteous", "irritable", "pretentious", "incurious",
    "unfathomable", "uncharitable", "malicious", "thoughtless", "crass", "hurried", "abrasive", "vague", "disorderly",
    "uncritical", "nihilistic", "fraudulent", "freewheeling", "frugal", "reserved", "graceless", "ruined", "suspicious",
    "oppressed", "hesitant", "stiff", "insulting", "unrealistic", "busy", "impressionable", "escapist", "fickle",
    "charmless", "disloyal", "mellow", "bland", "sly", "smooth", "old-fashioned", "predatory", "tense", "circumspect",
    "unreliable", "amusing", "arrogant", "scornful", "regretful", "paranoid", "provocative", "steely", "willful",
    "power-hungry", "ritualistic", "brittle", "pure", "dissonant", "preoccupied", "shallow", "arbitrary",
    "distractible", "soft", "narrow", "unrestrained", "disrespectful", "resentful", "unimpressive", "perverse",
    "flamboyant", "criminal", "casual", "opportunistic", "callous", "driving", "thievish", "obnoxious", "vindictive",
    "mechanical", "one-sided", "gloomy", "fixed", "asocial", "modern", "ridiculous", "argumentative", "ambitious",
    "devious", "rigid", "unreflective", "morbid", "apathetic", "cute", "unpolished", "outrageous", "discouraged",
    "unsentimental", "money-minded", "scheming", "neglectful", "moralistic", "absentminded", "offhand", "meddlesome",
    "progressive", "unctuous", "vulnerable", "forgetful", "abrupt", "disturbing", "idiosyncratic", "crude",
    "mannerless", "amoral"
]

# Occ_name_word
occ_name_targ1 = [
    "John",
    "Paul",
    "Mike",
    "Kevin",
    "Steve",
    "Greg",
    "Jeff",
    "Brad",
    "Brendan",
    "Geoffrey",
    "Brett",
    "Matthew",
    "Neil",
    "Darnell",
    "Hakim",
    "Jermaine",
    "Kareem",
    "Jamal",
    "Leroy",
    "Rasheed",
    "DeShawn",
    "DeAndre",
    "Marquis",
    "Terrell",
    "Malik",
    "Tyrone"
]
occ_name_targ1_reduced = ["Steve", "Mike", "Matthew", "Geoffrey", "Brendan", "Brett", "Tyrone", "Jamal", "Paul", "Jeff",
                          "Greg",
                          "Leroy", "Malik", "Marquis", "Kevin"]
occ_name_targ2 = [
    "Allison",
    "Anne",
    "Carrie",
    "Emily",
    "Jill",
    "Laurie",
    "Kristen",
    "Meredith",
    "Molly",
    "Amy",
    "Claire",
    "Katie",
    "Madeline",
    "Aisha",
    "Ebony",
    "Keisha",
    "Lakisha",
    "Latoya",
    "Tamika",
    "Imani",
    "Shanice",
    "Aaliyah",
    "Precious",
    "Nia",
    "Deja",
    "Latisha"
]
occ_name_targ2_reduced = ["Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie", "Kristen", "Meredith", "Molly", "Amy",
                          "Claire", "Katie", "Madeline", "Ebony", "Precious"]
occ_name_attr1 = [
    "driver",
    "supervisor",
    "janitor",
    "mover",
    "laborer",
    "construction",
    "worker",
    "chief",
    "developer",
    "carpenter",
    "manager",
    "lawyer",
    "farmer",
    "salesperson",
    "physician",
    "guard",
    "analyst",
    "mechanic",
    "sheriff",
    "ceo"
]
occ_name_attr2 = [
    "attendant",
    "cashier",
    "teacher",
    "nurse",
    "assistant",
    "secretary",
    "auditor",
    "cleaner",
    "receptionist",
    "clerk",
    "counselor",
    "designer",
    "hairdresser",
    "writer",
    "housekeeper",
    "baker",
    "accountant",
    "editor",
    "librarian",
    "tailor"
]

# Occ_term_word
occ_term_targ1 = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
occ_term_targ2 = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
occ_term_attr1 = [
    "driver",
    "supervisor",
    "janitor",
    "mover",
    "laborer",
    "construction",
    "worker",
    "chief",
    "developer",
    "carpenter",
    "manager",
    "lawyer",
    "farmer",
    "salesperson",
    "physician",
    "guard",
    "analyst",
    "mechanic",
    "sheriff",
    "ceo"
]
occ_term_attr2 = [
    "attendant",
    "cashier",
    "teacher",
    "nurse",
    "assistant",
    "secretary",
    "auditor",
    "cleaner",
    "receptionist",
    "clerk",
    "counselor",
    "designer",
    "hairdresser",
    "writer",
    "housekeeper",
    "baker",
    "accountant",
    "editor",
    "librarian",
    "tailor"
]

# I1_name_word
i1_name_targ1 = ["Andrew", "Brad", "Frank", "Geoffrey", "Jack", "Jonathan", "Josh", "Matthew", "Neil", "Peter", "Roger",
                 "Stephen"]
i1_name_targ2 = ["Aisha", "Keisha", "Lakisha", "Latisha", "Latoya", "Malika", "Nichelle", "Shereen", "Tamika",
                 "Tanisha",
                 "Yolanda", "Yvette"]
i1_name_attr1 = [
    "all-american",
    "arrogant",
    "attractive",
    "blond",
    "high-status",
    "intelligent",
    "leader",
    "privileged",
    "racist",
    "rich",
    "sexist",
    "sucessful",
    "tall"
]
i1_name_attr2 = [
    "aggressive",
    "athletic",
    "bigbutt",
    "confident",
    "darkskinned",
    "fried-chicken",
    "ghetto",
    "loud",
    "overweight",
    "promiscuous",
    "unfeminine",
    "unintelligent",
    "unrefined"
]

# I1_term_word
i1_term_targ1 = [
    "European American male",
    "British American male",
    "German American male",
    "Polish American male",
    "Russian American male",
    "Ukrainian American male",
    "Italian American male",
    "Portuguese American male",
    "French American male",
    "Romanian American male",
    "Greek American male",
    "Irish American male",
    "Spanish American male",
    "Bosnian American male",
    "Albanian American male",
    "European American man",
    "British American man",
    "German American man",
    "Polish American man",
    "Russian American man",
    "Ukrainian American man",
    "Italian American man",
    "Portuguese American man",
    "French American man",
    "Romanian American man",
    "Greek American man",
    "Irish American man",
    "Spanish American man",
    "Bosnian American man",
    "Albanian American man",
    "European American boy",
    "British American boy",
    "German American boy",
    "Polish American boy",
    "Russian American boy",
    "Ukrainian American boy",
    "Italian American boy",
    "Portuguese American boy",
    "French American boy",
    "Romanian American boy",
    "Greek American boy",
    "Irish American boy",
    "Spanish American boy",
    "Bosnian American boy",
    "Albanian American boy"
]
i1_term_targ2 = [
    "African American female",
    "Nigerian American female",
    "Ethiopian American female",
    "Egyptian American female",
    "Ghanaian American female",
    "Kenyan American female",
    "South African American female",
    "Somali American female",
    "Liberian American female",
    "Moroccan American female",
    "Cameroonian American female",
    "Cape Verdean American female",
    "Eritrean American female",
    "Sudanese American female",
    "Sierra Leonean American female",
    "African American woman",
    "Nigerian American woman",
    "Ethiopian American woman",
    "Egyptian American woman",
    "Ghanaian American woman",
    "Kenyan American woman",
    "South African American woman",
    "Somali American woman",
    "Liberian American woman",
    "Moroccan American woman",
    "Cameroonian American woman",
    "Cape Verdean American woman",
    "Eritrean American woman",
    "Sudanese American woman",
    "Sierra Leonean American woman",
    "African American girl",
    "Nigerian American girl",
    "Ethiopian American girl",
    "Egyptian American girl",
    "Ghanaian American girl",
    "Kenyan American girl",
    "South African American girl",
    "Somali American girl",
    "Liberian American girl",
    "Moroccan American girl",
    "Cameroonian American girl",
    "Cape Verdean American girl",
    "Eritrean American girl",
    "Sudanese American girl",
    "Sierra Leonean American girl"
]
i1_term_attr1 = [
    "all-american",
    "arrogant",
    "attractive",
    "blond",
    "high-status",
    "intelligent",
    "leader",
    "privileged",
    "racist",
    "rich",
    "sexist",
    "sucessful",
    "tall"
]
i1_term_attr2 = [
    "aggressive",
    "athletic",
    "bigbutt",
    "confident",
    "darkskinned",
    "fried-chicken",
    "ghetto",
    "loud",
    "overweight",
    "promiscuous",
    "unfeminine",
    "unintelligent",
    "unrefined"
]

# I2_name_word
i2_name_targ1 = ["Andrew", "Brad", "Frank", "Geoffrey", "Jack", "Jonathan", "Josh", "Matthew", "Neil", "Peter", "Roger",
                 "Stephen"]
i2_name_targ2 = ["Aisha", "Keisha", "Lakisha", "Latisha", "Latoya", "Malika", "Nichelle", "Shereen", "Tamika",
                 "Tanisha",
                 "Yolanda", "Yvette"]
i2_name_attr1 = ["arrogant", "blond", "high-status", "intelligent", "racist", "rich", "sucessful", "tall"]
i2_name_attr2 = ["aggressive", "bigbutt", "confident", "darkskinned", "fried-chicken", "overweight", "promiscuous",
                 "unfeminine"]

# I2_term_word
i2_term_targ1 = [
    "European American male",
    "British American male",
    "German American male",
    "Polish American male",
    "Russian American male",
    "Ukrainian American male",
    "Italian American male",
    "Portuguese American male",
    "French American male",
    "Romanian American male",
    "Greek American male",
    "Irish American male",
    "Spanish American male",
    "Bosnian American male",
    "Albanian American male",
    "European American man",
    "British American man",
    "German American man",
    "Polish American man",
    "Russian American man",
    "Ukrainian American man",
    "Italian American man",
    "Portuguese American man",
    "French American man",
    "Romanian American man",
    "Greek American man",
    "Irish American man",
    "Spanish American man",
    "Bosnian American man",
    "Albanian American man",
    "European American boy",
    "British American boy",
    "German American boy",
    "Polish American boy",
    "Russian American boy",
    "Ukrainian American boy",
    "Italian American boy",
    "Portuguese American boy",
    "French American boy",
    "Romanian American boy",
    "Greek American boy",
    "Irish American boy",
    "Spanish American boy",
    "Bosnian American boy",
    "Albanian American boy"
]
i2_term_targ2 = [
    "African American female",
    "Nigerian American female",
    "Ethiopian American female",
    "Egyptian American female",
    "Ghanaian American female",
    "Kenyan American female",
    "South African American female",
    "Somali American female",
    "Liberian American female",
    "Moroccan American female",
    "Cameroonian American female",
    "Cape Verdean American female",
    "Eritrean American female",
    "Sudanese American female",
    "Sierra Leonean American female",
    "African American woman",
    "Nigerian American woman",
    "Ethiopian American woman",
    "Egyptian American woman",
    "Ghanaian American woman",
    "Kenyan American woman",
    "South African American woman",
    "Somali American woman",
    "Liberian American woman",
    "Moroccan American woman",
    "Cameroonian American woman",
    "Cape Verdean American woman",
    "Eritrean American woman",
    "Sudanese American woman",
    "Sierra Leonean American woman",
    "African American girl",
    "Nigerian American girl",
    "Ethiopian American girl",
    "Egyptian American girl",
    "Ghanaian American girl",
    "Kenyan American girl",
    "South African American girl",
    "Somali American girl",
    "Liberian American girl",
    "Moroccan American girl",
    "Cameroonian American girl",
    "Cape Verdean American girl",
    "Eritrean American girl",
    "Sudanese American girl",
    "Sierra Leonean American girl"
]
i2_term_attr1 = ["arrogant", "blond", "high-status", "intelligent", "racist", "rich", "sucessful", "tall"]
i2_term_attr2 = ["aggressive", "bigbutt", "confident", "darkskinned", "fried-chicken", "overweight", "promiscuous",
                 "unfeminine"]

def get_stimuli(test_name, reduced_wd_sets):
    """ Function to get stimuli for specified bias test """
    if reduced_wd_sets:
        if test_name == 'c1_name':
            targ1, targ2, attr1, attr2 = c1_name_targ1_reduced, c1_name_targ2_reduced, c1_name_attr1, c1_name_attr2
        elif test_name == 'c3_name':
            targ1, targ2, attr1, attr2 = c3_name_targ1_reduced, c3_name_targ2_reduced, c3_name_attr1, c3_name_attr2
        elif test_name == 'c9_name':
            targ1, targ2, attr1, attr2 = c9_name_targ1_reduced, c9_name_targ2_reduced, c9_name_attr1, c9_name_attr2
        elif test_name == 'c9_name_m':
            targ1, targ2, attr1, attr2 = c9_name_m_targ1_reduced, c9_name_m_targ2_reduced, c9_name_m_attr1, c9_name_m_attr2
        elif test_name == 'c9_term':
            targ1, targ2, attr1, attr2 = c9_term_targ1_reduced, c9_term_targ2_reduced, c9_term_attr1, c9_term_attr2
        elif test_name == 'occ_name':
            targ1, targ2, attr1, attr2 = occ_name_targ1_reduced, occ_name_targ2_reduced, occ_name_attr1, occ_name_attr2
        else:
            raise ValueError("Reduced dataset for bias test %s not found!" % test_name)
    else:
        if test_name == 'c1_name':
            targ1, targ2, attr1, attr2 = c1_name_targ1, c1_name_targ2, c1_name_attr1, c1_name_attr2
        elif test_name == 'c3_name':
            targ1, targ2, attr1, attr2 = c3_name_targ1, c3_name_targ2, c3_name_attr1, c3_name_attr2
        elif test_name == 'c3_term':
            targ1, targ2, attr1, attr2 = c3_term_targ1, c3_term_targ2, c3_term_attr1, c3_term_attr2
        elif test_name == 'c6_name':
            targ1, targ2, attr1, attr2 = c6_name_targ1, c6_name_targ2, c6_name_attr1, c6_name_attr2
        elif test_name == 'c6_term':
            targ1, targ2, attr1, attr2 = c6_term_targ1, c6_term_targ2, c6_term_attr1, c6_term_attr2
        elif test_name == 'c9_name':
            targ1, targ2, attr1, attr2 = c9_name_targ1, c9_name_targ2, c9_name_attr1, c9_name_attr2
        elif test_name == 'c9_name_m':
            targ1, targ2, attr1, attr2 = c9_name_m_targ1, c9_name_m_targ2, c9_name_m_attr1, c9_name_m_attr2
        elif test_name == 'c9_term':
            targ1, targ2, attr1, attr2 = c9_term_targ1, c9_term_targ2, c9_term_attr1, c9_term_attr2
        elif test_name == 'dis_term':
            targ1, targ2, attr1, attr2 = dis_term_targ1, dis_term_targ2, dis_term_attr1, dis_term_attr2
        elif test_name == 'dis_term_m':
            targ1, targ2, attr1, attr2 = dis_term_m_targ1, dis_term_m_targ2, dis_term_m_attr1, dis_term_m_attr2
        elif test_name == 'occ_name':
            targ1, targ2, attr1, attr2 = occ_name_targ1, occ_name_targ2, occ_name_attr1, occ_name_attr2
        elif test_name == 'occ_term':
            targ1, targ2, attr1, attr2 = occ_term_targ1, occ_term_targ2, occ_term_attr1, occ_term_attr2
        elif test_name == 'i1_name':
            targ1, targ2, attr1, attr2 = i1_name_targ1, i1_name_targ2, i1_name_attr1, i1_name_attr2
        elif test_name == 'i1_term':
            targ1, targ2, attr1, attr2 = i1_term_targ1, i1_term_targ2, i1_term_attr1, i1_term_attr2
        elif test_name == 'i2_name':
            targ1, targ2, attr1, attr2 = i2_name_targ1, i2_name_targ2, i2_name_attr1, i2_name_attr2
        elif test_name == 'i2_term':
            targ1, targ2, attr1, attr2 = i2_term_targ1, i2_term_targ2, i2_term_attr1, i2_term_attr2
        else:
            raise ValueError("Stimuli for bias test %s not found!" % test_name)

    return targ1, targ2, attr1, attr2

def filter_sent(attr, targ, sent_dict):
    """ Function to filter and re-fill dict for attribute words (keys) """
    sents = {attr_wd: {targ_wd: None for targ_wd in targ} for attr_wd in attr}
    for key, value in sent_dict.items():
        for key_tuple, value_tuple in value.items():
            if key_tuple[1] in attr:
                sents[key_tuple[1]][key_tuple[0]] = value_tuple
    return sents

def shorten_sent(sent, targ_wd, attr_wd):
    """ Function to shorten the raw comment
    if applicable take window of size 20 (average sentence length)
    """
    window_size = 50
    wds = sent.split()

    if len(wds) > window_size:  # case: comment longer than 20 words

        len_targ, len_attr = len(targ_wd.split()), len(attr_wd.split())
        len_max = max(len_targ, len_attr)  # account for len of longest word of interest

        # determine idxs of target and attribute word
        flag_targ, flag_attr = False, False
        if len_targ > 1:
              idx_charac = ' '.join(wds).find(targ_wd)
              i, len_charac = 0, 0
              for element in wds:
                    if len_charac == idx_charac:
                          idx_targ = i
                          flag_targ = True
                    i += 1
                    len_charac += len(element) + 1
        else:
              idx_targ = wds.index(targ_wd)
              flag_targ = True
        if len_attr > 1:
              idx_charac = ' '.join(wds).find(attr_wd)
              i, len_charac = 0, 0
              for element in wds:
                    if len_charac == idx_charac:
                          idx_attr = i
                          flag_attr = True
                    i += 1
                    len_charac += len(element) + 1
        else:
              idx_attr = wds.index(attr_wd)
              flag_attr = True

        if flag_targ and flag_attr: # case: successful mapping of idxs of target and attribute word

            idx_smaller = min(idx_targ, idx_attr)
            idx_larger = max(idx_targ, idx_attr)

            # case: less than 18 words between start of words of interest
            if abs(idx_targ - idx_attr) <= (window_size - 1):
                dist_wds = abs(idx_targ - idx_attr)
                pad_size = int((window_size - dist_wds) / 2)

                # case: take first window_size words
                if idx_smaller < pad_size:
                    idx_end = window_size + 2 * len_max # account for len of longest word of interest
                    new_sent = ' '.join(wds[:idx_end])
                # case: take last window_size words
                elif (len(wds) - 1 - idx_larger - len_max) < pad_size:
                    idx_end = window_size + 2 * len_max # account for len of longest word of interest
                    new_sent = ' '.join(wds[-idx_end:])
                # case take pad_size words before and after stimuli
                else:
                    idx_start = idx_smaller - pad_size
                    idx_end = idx_start + window_size + 2 * len_max
                    new_sent = ' '.join(wds[idx_start:idx_end])

            # case: more than 18 words between start of words of interest
            else:
                new_sent = ''

        else: # case: no successful mapping of idxs of target and attribute word
              new_sent = ''

    else: # case: comment shorter than 20 words
        new_sent = sent

    return new_sent

def load_model(model_name):
    """ Load model and corresponding tokenizers if applicable """
    if model_name == 'bert':
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # additional 'Fast' BERT tokenizer for subword tokenization ID mapping
        subword_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    else:
        raise ValueError("Model %s not found!" % model_name)

    return model.to(device), tokenizer, subword_tokenizer

def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)

def bert_logits(sent_tuple, sent, attr_wd, targ1_wd, targ2_wd, model, tok, subword_tok, apply_softmax=True):
    """ Function to obtain (prior) logits for BERT """

    encodings = tok(sent, return_tensors='pt', truncation=True)  # [CLS] and [SEP] tokens are added automatically
    #token_ids = torch.tensor(encodings['input_ids'], device=device)
    token_ids = encodings['input_ids'].clone().detach().to(device)
    # map tokens to input words
    subword_ids = subword_tok(sent, add_special_tokens=False).word_ids()
    outputs = model(input_ids=token_ids)
    logits = outputs.logits[0, :, :].cpu().detach().numpy()
    if apply_softmax:
        logits = softmax(logits)

    # determine idx of [MASK] in input sentence
    # case: AAA occurs before TTT and thus take second / last occurrence of [MASK] token
    if sent.count('[MASK]') > 1 and sent_tuple[1].find(attr_wd) < sent_tuple[1].find(sent_tuple[0]):
        idx_mask = None
        tokens = sent.split()
        for i, token in enumerate(tokens):
            if token == '[MASK]':
                idx_mask = i
    # case: TTT occurs before AAA and thus take first occurrence of [MASK] token
    else:
        idx_mask = sent.split().index('[MASK]')

    # unknown bug - try to catch and thus omit sent
    try:
        # case: subword tokenization before or after [MASK] token
        if len(sent.split()) != len(subword_ids):
            idx_mask = [i for i in range(len(subword_ids)) if subword_ids[i] == idx_mask][0] + 1  # account for CLS token
        # case: no subword tokenization in input sentence
        else:
            idx_mask += 1  # account for CLS token

        result = {targ1_wd: float(), targ2_wd: float()}
        for targ_wd in [targ1_wd.lower(), targ2_wd.lower()]:
            # case: target word consists of subwords or multiple words
            if len(subword_tok(targ_wd, add_special_tokens=False).word_ids()) > 1:
                subword_token_ids = subword_tok(targ_wd, add_special_tokens=False)['input_ids']
                subwords = [k for k, v in tok.vocab.items() if v in subword_token_ids]
                subwords_logits = [logits[idx_mask, tok.vocab[subwd]] for subwd in subwords]
                result[targ_wd] = np.prod(subwords_logits)  # take prod of all probs
            else:
                result[targ_wd] = logits[idx_mask, tok.vocab[targ_wd]]

        return result

    except:
        result = {targ1_wd: float(), targ2_wd: float()}
        for targ_wd in [targ1_wd.lower(), targ2_wd.lower()]:
            result[targ_wd] = np.nan
        return result

def bias_score(sent_tuple, attr_wd, targ1_wds, targ2_wds, model, tok, subword_tok):
    """ Function to compute the log prob bias score
        args:
        - sent_tuple (tuple): (target word in sentence, sentence wrt attribute word of interest)
        - attr_wd (str): attribute word of interest
        - targ1_wds, targ2_wds (list): list containing respective target words
        - model, tok, subword_tok: specified LM including tokenizers
    """

    sent_short = shorten_sent(sent_tuple[1], sent_tuple[0], attr_wd)

    if sent_short != '': # case: sentence could successfully be shortened
         if sent_tuple[0] in targ1_wds:
             targ1_wd = sent_tuple[0]
             targ2_wd = random.choice(targ2_wds) # random sample counterpart
         elif sent_tuple[0] in targ2_wds:
             targ2_wd = sent_tuple[0]
             targ1_wd = random.choice(targ1_wds) # random sample counterpart
         else:
             raise ValueError("Stimuli %s not found!" % sent_tuple[0])

         targ_wd = sent_tuple[0]
         mask_token_targ = '[MASK]'
         bias = np.nan
         bias_prior_correction = np.nan

         sent_replaced = sent_short.replace(targ_wd, mask_token_targ)
         if '[MASK]' in sent_replaced.split():
             # p_tgt : prob of filling [MASK] token with target words given sent with attribute word
            logits_tgt = bert_logits(sent_tuple, sent_replaced,
                                     attr_wd, targ1_wd, targ2_wd, model, tok, subword_tok)
            bias = np.log(logits_tgt[targ1_wd]) - np.log(logits_tgt[targ2_wd])
            # p_prior : prob of filling [MASK] token with target words given sent with masked attribute word
            logits_prior = bert_logits(sent_tuple, sent_replaced.replace(attr_wd, '[MASK]'),
                                       attr_wd, targ1_wd, targ2_wd, model, tok, subword_tok)
            bias_prior_correction = np.log(logits_prior[targ1_wd]) - np.log(logits_prior[targ2_wd])

         return {"stimulus": attr_wd,
                 "bias": bias,
                 "prior_correction": bias_prior_correction,
                 "bias_prior_corrected": bias - bias_prior_correction}

    else: # case: sentence could not successfully be shortened
          return {"stimulus": attr_wd,
                  "bias": np.nan,
                  "prior_correction": np.nan,
                  "bias_prior_corrected": np.nan}

def exact_perm_test(xs, ys, nmc=100000):
    """ Function to compute p-value """
    n, k = len(xs), 0
    s = np.abs(np.mean(xs) - np.mean(ys))  # two-sided p-value
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)  # permutation test
        k += s < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

def logprob_cal(model, tokenizer, subword_tokenizer, sents_attr1, sents_attr2, targ1, targ2):
    """ Function to run log prob bias score calculations
        args:
            - model, tokenizer, subword_tokenizer: specified LM including tokenizers
            - sents_attr1 (Dict[str: Dict]): dictionary mapping sentences containing specific target stimuli to attribute words from A
            - sents_attr2 (Dict[str: Dict]): dictionary mapping sentences containing specific target stimuli to attribute words from B
    """
    df_attr1_lst, df_attr2_lst = [], []
    for attr_wd in sents_attr1.keys():
        sents = []
        for k, v in sents_attr1[attr_wd].items():
            for sent in v:
                sents.append((k, sent))
        results = [bias_score(sent_tuple, attr_wd, targ1, targ2, model, tokenizer, subword_tokenizer)
                   for sent_tuple in sents]
        df_attr1_lst.append(pd.DataFrame(results))
    for attr_wd in sents_attr2.keys():
        sents = []
        for k, v in sents_attr2[attr_wd].items():
            for sent in v:
                sents.append((k, sent))
        results = [bias_score(sent_tuple, attr_wd, targ1, targ2, model, tokenizer, subword_tokenizer)
                   for sent_tuple in sents]
        df_attr2_lst.append(pd.DataFrame(results))
    df1 = pd.concat(df_attr1_lst).replace([np.inf, -np.inf], np.nan).dropna()
    df2 = pd.concat(df_attr2_lst).replace([np.inf, -np.inf], np.nan).dropna()

    k = 'bias_prior_corrected'
    std_AB = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
    esize = (df1[k].mean() - df2[k].mean()) / std_AB
    pvalue = exact_perm_test(df1[k], df2[k])

    return esize, pvalue, df1.shape[0], df2.shape[0]


sent_dict = pickle.load(open('sent_dict_double.pickle','rb'))

all_tests = ['c1_name', 'c3_name', 'c3_term', 'c6_name', 'c6_term', 'c9_name', 'c9_name_m', 'c9_term',
             'occ_name', 'occ_term', 'dis_term', 'dis_term_m', 'i1_name', 'i1_term', 'i2_name', 'i2_term']

reduced_tests = ['c1_name', 'c3_name', 'c9_name', 'c9_name_m', 'c9_term', 'occ_name']
# for c6_name, c6_term, occ_term the reduced word sets did not change compared to the original word sets
# for c3_term, dis_term, dis_term_m, i1_name, i1_term , i2_name, i2_term the word sets reduced to 0 stimuli

relevant_tests = ['c1_name', 'c3_name', 'c6_name', 'c9_term', 'dis_term', 'occ_name', 'i1_name', 'i2_name']
# for conference paper only specific tests are needed

results = []

for test in relevant_tests:
    reduced_wd_sets = False
    targ1, targ2, attr1, attr2 = get_stimuli(test, reduced_wd_sets)

    targ = targ1 + targ2
    sent_dict_small = {k: v for k, v in sent_dict.items() if k in targ}

    sents_attr1 = filter_sent(attr1, targ, sent_dict_small)
    sents_attr2 = filter_sent(attr2, targ, sent_dict_small)

    bert_model, bert_tok, bert_sub_tok = load_model('bert')

    print(f'Starting to compute logits for bias test {test}')
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    esize, pvalue, len_df1, len_df2 = logprob_cal(bert_model, bert_tok, bert_sub_tok, sents_attr1, sents_attr2, targ1, targ2)

    results.append(dict(
        method='LPBS',
        test=test,
        model='bert',
        dataset='full',
        evaluation_measure='probability',
        context='reddit',
        encoding_level='',
        p_value=pvalue,
        effect_size=esize,
        num_df1=len_df1,
        num_df2=len_df2))

# for test in reduced_tests:
#      reduced_wd_sets = True
#      ...

# save results and specs of code run (time, date)
results_path = time.strftime("%Y%m%d-%H%M%S") + '_LPBS_bert_reddit.csv'
print('Writing results to {}'.format(results_path))
with open(results_path, 'w') as f:
    writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
    writer.writeheader()
    for r in results:
        writer.writerow(r)
