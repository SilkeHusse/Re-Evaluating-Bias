""" Generate BERT embeddings, execute CEAT method, and save results """
import pickle
import datetime
import time
import math
import torch
import scipy.stats
import numpy as np
import random

random.seed(1111)

from csv import DictWriter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizerFast

if torch.cuda.is_available():
    print('GPU is available.')
    device = torch.device("cuda") # set device
else:
    print('No GPU available, using CPU instead.')
    device = torch.device("cpu")

# C1_name_word
c1_name_targ1 = ["aster","clover","hyacinth","marigold","poppy","azalea","crocus","iris","orchid","rose","bluebell",
                 "daffodil","lilac","pansy","tulip","buttercup","daisy","lily","peony","violet","carnation","gladiola",
                 "magnolia","petunia","zinnia"]
c1_name_targ2 = ["ant","caterpillar","flea","locust","spider","bedbug","centipede","fly","maggot","tarantula","bee",
                 "cockroach","gnat","mosquito","termite","beetle","cricket","hornet","moth","wasp","blackfly",
                 "dragonfly","horsefly","roach","weevil"]
c1_name_attr1 = ["caress","freedom","health","love","peace","cheer","friend","heaven","loyal","pleasure","diamond",
                 "gentle","honest","lucky","rainbow","diploma","gift","honor","miracle","sunrise","family","happy",
                 "laughter","paradise","vacation"]
c1_name_attr2 = ["abuse","crash","filth","murder","sickness","accident","death","grief","poison","stink","assault",
                 "disaster","hatred","pollute","tragedy","divorce","jail","poverty","ugly","cancer","kill","rotten",
                 "vomit","agony","prison"]

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
c3_name_attr1 = ["caress","freedom","health","love","peace","cheer","friend","heaven","loyal","pleasure","diamond",
                 "gentle","honest","lucky","rainbow","diploma","gift","honor","miracle","sunrise","family","happy",
                 "laughter","paradise","vacation"]
c3_name_attr2 = ["abuse","crash","filth","murder","sickness","accident","death","grief","poison","stink","assault",
                 "disaster","hatred","pollute","tragedy","bomb","divorce","jail","poverty","ugly","cancer","evil",
                 "kill","rotten","vomit"]

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
c3_term_attr1 = ["caress","freedom","health","love","peace","cheer","friend","heaven","loyal","pleasure","diamond",
                 "gentle","honest","lucky","rainbow","diploma","gift","honor","miracle","sunrise","family","happy",
                 "laughter","paradise","vacation"]
c3_term_attr2 = ["abuse","crash","filth","murder","sickness","accident","death","grief","poison","stink","assault",
                 "disaster","hatred","pollute","tragedy","bomb","divorce","jail","poverty","ugly","cancer","evil",
                 "kill","rotten","vomit"]

# C6_name_word
c6_name_targ1 = ["John","Paul","Mike","Kevin","Steve","Greg","Jeff","Bill"]
c6_name_targ2 = ["Amy","Joan","Lisa","Sarah","Diana","Kate","Ann","Donna"]
c6_name_attr1 = ["executive","management","professional","corporation","salary","office","business","career"]
c6_name_attr2 = ["home","parent","child","family","cousin","marriage","wedding","relative"]

# C6_term_word
c6_term_targ1 = ["male","man","boy","brother","he","him","his","son"]
c6_term_targ2 = ["female","woman","girl","sister","she","her","hers","daughter"]
c6_term_attr1 = ["executive","management","professional","corporation","salary","office","business","career"]
c6_term_attr2 = ["home","parent","child","family","cousin","marriage","wedding","relative"]

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
c9_name_attr1 = ["impermanent","unstable","variable","fleeting","short-term","brief","occasional"]
c9_name_attr2 = ["stable","always","constant","persistent","chronic","prolonged","forever"]

# C9m_name_word
c9m_name_targ1 = [
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
c9m_name_targ2 = [
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
c9m_name_attr1 = ["impermanent","unstable","variable","fleeting","short-term","brief","occasional","transitory"]
c9m_name_attr2 = ["stable","always","constant","persistent","chronic","prolonged","forever","lasting"]

# C9_term_word
c9_term_targ1 = ["sad","hopeless","gloomy","tearful","miserable","depressed"]
c9_term_targ2 = ["sick","illness","influenza","disease","virus","cancer"]
c9_term_attr1 = ["impermanent","unstable","variable","fleeting","short-term","brief","occasional"]
c9_term_attr2 = ["stable","always","constant","persistent","chronic","prolonged","forever"]

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
      "accessible", "active", "adaptable", "admirable", "adventurous", "agreeable", "alert", "amiable", "appreciative", "articulate", "aspiring", "athletic", "balanced", "benevolent", "brilliant", "calm", "capable", "captivating", "caring", "challenging", "charismatic", "charming", "cheerful", "clean", "clear-headed", "clever", "companionly", "compassionate", "conciliatory", "confident", "conscientious", "considerate", "constant", "contemplative", "cooperative", "courageous", "courteous", "creative", "cultured", "curious", "daring", "debonair", "decent", "dedicated", "deep", "dignified", "directed", "disciplined", "discreet", "dramatic", "dutiful", "dynamic", "earnest", "ebullient", "educated", "efficient", "elegant", "eloquent", "empathetic", "energetic", "enthusiastic", "esthetic", "exciting", "extraordinary", "fair", "faithful", "farsighted", "felicific", "firm", "flexible", "focused", "forceful", "forgiving", "forthright", "freethinking", "friendly", "fun-loving", "gallant", "generous", "gentle", "genuine", "good-natured", "hardworking", "healthy", "hearty", "helpful", "heroic", "high-minded", "honest", "honorable", "humble", "humorous", "idealistic", "imaginative", "impressive", "incisive", "incorruptible", "independent", "individualistic", "innovative", "inoffensive", "insightful", "insouciant", "intelligent", "intuitive", "invulnerable", "kind", "knowledge", "leader", "leisurely", "liberal", "logical", "lovable", "loyal", "lyrical", "magnanimous", "many-sided", "masculine", "mature", "methodical", "meticulous", "moderate", "modest", "multi-leveled", "neat", "objective", "observant", "open", "optimistic", "orderly", "organized", "original", "painstaking", "passionate", "patient", "patriotic", "peaceful", "perceptive", "perfectionist", "personable", "persuasive", "playful", "polished", "popular", "practical", "precise", "principled", "profound", "protean", "protective", "providential", "prudent", "punctual", "purposeful", "rational", "realistic", "reflective", "relaxed", "reliable", "resourceful", "respectful", "responsible", "responsive", "reverential", "romantic", "rustic", "sage", "sane", "scholarly", "scrupulous", "secure", "selfless", "self-critical", "self-defacing", "self-denying", "self-reliant", "self-sufficent", "sensitive", "sentimental", "seraphic", "serious", "sexy", "sharing", "shrewd", "simple", "skillful", "sober", "sociable", "solid", "sophisticated", "spontaneous", "sporting", "stable", "steadfast", "steady", "stoic", "strong", "studious", "suave", "subtle", "sweet", "sympathetic", "systematic", "tasteful", "teacherly", "thorough", "tidy", "tolerant", "tractable", "trusting", "uncomplaining", "understanding", "undogmatic", "upright", "urbane", "venturesome", "vivacious", "warm", "well-bred", "well-read", "well-rounded", "winning", "wise", "witty", "youthful"
      ]
dis_term_attr2 = [
      "contradictory", "envious", "conformist", "frightening", "experimental", "gullible", "careless", "impulsive", "skeptical", "big-thinking", "dreamy", "angry", "undisciplined", "miserable", "haughty", "familial", "unimaginative", "mystical", "ungrateful", "maternal", "prim", "fearful", "submissive", "insecure", "colorless", "competitive", "passive", "superficial", "destructive", "impatient", "brutal", "aggressive", "ignorant", "placid", "chummy", "cerebral", "unlovable", "desperate", "pedantic", "hedonistic", "compulsive", "cowardly", "unfriendly", "regimental", "stylish", "invisible", "greedy", "foolish", "indecisive", "procrastinating", "outspoken", "clumsy", "imitative", "retiring", "irreverent", "unambitious", "physical", "folksy", "irresponsible", "slow", "softheaded", "intense", "sensual", "petty", "effeminate", "transparent", "sedentary", "dull", "uninhibited", "erratic", "pompous", "discontented", "vacuous", "delicate", "narcissistic", "irreligious", "irrational", "muddle-headed", "sanctimonious", "frivolous", "fawning", "whimsical", "tactless", "unstable", "unprincipled", "agonizing", "proud", "questioning", "irascible", "false", "conceited", "hypnotic", "artificial", "calculating", "cantankerous", "surprising", "contemptible", "emotional", "condemnatory", "subjective", "political", "unpredictable", "insincere", "conventional", "unceremonious", "treacherous", "religious", "quirky", "boyish", "complaining", "undemanding", "authoritarian", "disobedient", "disputatious", "discourteous", "irritable", "pretentious", "incurious", "unfathomable", "uncharitable", "malicious", "thoughtless", "crass", "hurried", "abrasive", "vague", "disorderly", "uncritical", "nihilistic", "fraudulent", "freewheeling", "frugal", "reserved", "graceless", "ruined", "suspicious", "oppressed", "hesitant", "stiff", "insulting", "unrealistic", "busy", "impressionable", "escapist", "fickle", "charmless", "disloyal", "mellow", "bland", "sly", "smooth", "old-fashioned", "predatory", "tense", "circumspect", "unreliable", "amusing", "arrogant", "scornful", "regretful", "paranoid", "provocative", "steely", "willful", "power-hungry", "ritualistic", "brittle", "pure", "dissonant", "preoccupied", "shallow", "arbitrary", "distractible", "soft", "narrow", "unrestrained", "disrespectful", "resentful", "unimpressive", "perverse", "flamboyant", "criminal", "casual", "opportunistic", "callous", "driving", "thievish", "obnoxious", "vindictive", "mechanical", "one-sided", "gloomy", "fixed", "asocial", "modern", "ridiculous", "argumentative", "ambitious", "devious", "rigid", "unreflective", "morbid", "apathetic", "cute", "unpolished", "outrageous", "discouraged", "unsentimental", "money-minded", "scheming", "neglectful", "moralistic", "absentminded", "offhand", "meddlesome", "progressive", "unctuous", "vulnerable", "forgetful", "abrupt", "disturbing", "idiosyncratic", "crude", "mannerless", "amoral"
    ]

# Dism_term_word
dism_term_targ1 = [
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
dism_term_targ2 = [
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
dism_term_attr1 = [
      "accessible", "active", "adaptable", "admirable", "adventurous", "agreeable", "alert", "amiable", "appreciative", "articulate", "aspiring", "athletic", "balanced", "benevolent", "brilliant", "calm", "capable", "captivating", "caring", "challenging", "charismatic", "charming", "cheerful", "clean", "clear-headed", "clever", "companionly", "compassionate", "conciliatory", "confident", "conscientious", "considerate", "constant", "contemplative", "cooperative", "courageous", "courteous", "creative", "cultured", "curious", "daring", "debonair", "decent", "dedicated", "deep", "dignified", "directed", "disciplined", "discreet", "dramatic", "dutiful", "dynamic", "earnest", "ebullient", "educated", "efficient", "elegant", "eloquent", "empathetic", "energetic", "enthusiastic", "esthetic", "exciting", "extraordinary", "fair", "faithful", "farsighted", "felicific", "firm", "flexible", "focused", "forceful", "forgiving", "forthright", "freethinking", "friendly", "fun-loving", "gallant", "generous", "gentle", "genuine", "good-natured", "hardworking", "healthy", "hearty", "helpful", "heroic", "high-minded", "honest", "honorable", "humble", "humorous", "idealistic", "imaginative", "impressive", "incisive", "incorruptible", "independent", "individualistic", "innovative", "inoffensive", "insightful", "insouciant", "intelligent", "intuitive", "invulnerable", "kind", "knowledge", "leader", "leisurely", "liberal", "logical", "lovable", "loyal", "lyrical", "magnanimous", "many-sided", "mature", "methodical", "meticulous", "moderate", "modest", "multi-leveled", "neat", "objective", "observant", "open", "optimistic", "orderly", "organized", "original", "painstaking", "passionate", "patient", "patriotic", "peaceful", "perceptive", "perfectionist", "personable", "persuasive", "playful", "polished", "popular", "practical", "precise", "principled", "profound", "protean", "protective", "providential", "prudent", "punctual", "purposeful", "rational", "realistic", "reflective", "relaxed", "reliable", "resourceful", "respectful", "responsible", "responsive", "reverential", "romantic", "rustic", "sage", "sane", "scholarly", "scrupulous", "secure", "selfless", "self-critical", "self-defacing", "self-denying", "self-reliant", "self-sufficent", "sensitive", "sentimental", "seraphic", "serious", "sexy", "sharing", "shrewd", "simple", "skillful", "sober", "sociable", "solid", "sophisticated", "spontaneous", "sporting", "stable", "steadfast", "steady", "stoic", "strong", "studious", "suave", "subtle", "sweet", "sympathetic", "systematic", "tasteful", "teacherly", "thorough", "tidy", "tolerant", "tractable", "trusting", "uncomplaining", "understanding", "undogmatic", "upright", "urbane", "venturesome", "vivacious", "warm", "well-bred", "well-read", "well-rounded", "winning", "wise", "witty"
      ]
dism_term_attr2 = [
      "contradictory", "envious", "conformist", "frightening", "experimental", "gullible", "careless", "impulsive", "skeptical", "big-thinking", "dreamy", "angry", "undisciplined", "miserable", "haughty", "familial", "unimaginative", "mystical", "ungrateful", "prim", "fearful", "submissive", "insecure", "colorless", "competitive", "passive", "superficial", "destructive", "impatient", "brutal", "aggressive", "ignorant", "placid", "chummy", "cerebral", "unlovable", "desperate", "pedantic", "hedonistic", "compulsive", "cowardly", "unfriendly", "regimental", "stylish", "invisible", "greedy", "foolish", "indecisive", "procrastinating", "outspoken", "clumsy", "imitative", "retiring", "irreverent", "unambitious", "physical", "folksy", "irresponsible", "slow", "softheaded", "intense", "sensual", "petty", "effeminate", "transparent", "sedentary", "dull", "uninhibited", "erratic", "pompous", "discontented", "vacuous", "delicate", "narcissistic", "irreligious", "irrational", "muddle-headed", "sanctimonious", "frivolous", "fawning", "whimsical", "tactless", "unstable", "unprincipled", "agonizing", "proud", "questioning", "irascible", "false", "conceited", "hypnotic", "artificial", "calculating", "cantankerous", "surprising", "contemptible", "emotional", "condemnatory", "subjective", "political", "unpredictable", "insincere", "conventional", "unceremonious", "treacherous", "religious", "quirky", "complaining", "undemanding", "authoritarian", "disobedient", "disputatious", "discourteous", "irritable", "pretentious", "incurious", "unfathomable", "uncharitable", "malicious", "thoughtless", "crass", "hurried", "abrasive", "vague", "disorderly", "uncritical", "nihilistic", "fraudulent", "freewheeling", "frugal", "reserved", "graceless", "ruined", "suspicious", "oppressed", "hesitant", "stiff", "insulting", "unrealistic", "busy", "impressionable", "escapist", "fickle", "charmless", "disloyal", "mellow", "bland", "sly", "smooth", "old-fashioned", "predatory", "tense", "circumspect", "unreliable", "amusing", "arrogant", "scornful", "regretful", "paranoid", "provocative", "steely", "willful", "power-hungry", "ritualistic", "brittle", "pure", "dissonant", "preoccupied", "shallow", "arbitrary", "distractible", "soft", "narrow", "unrestrained", "disrespectful", "resentful", "unimpressive", "perverse", "flamboyant", "criminal", "casual", "opportunistic", "callous", "driving", "thievish", "obnoxious", "vindictive", "mechanical", "one-sided", "gloomy", "fixed", "asocial", "modern", "ridiculous", "argumentative", "ambitious", "devious", "rigid", "unreflective", "morbid", "apathetic", "cute", "unpolished", "outrageous", "discouraged", "unsentimental", "money-minded", "scheming", "neglectful", "moralistic", "absentminded", "offhand", "meddlesome", "progressive", "unctuous", "vulnerable", "forgetful", "abrupt", "disturbing", "idiosyncratic", "crude", "mannerless", "amoral"
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
occ_term_targ1 = ["male","man","boy","brother","he","him","his","son"]
occ_term_targ2 = ["female","woman","girl","sister","she","her","hers","daughter"]
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

# IBD_name_word
ibd_name_targ1 = ["Andrew","Brad","Frank","Geoffrey","Jack","Jonathan","Josh","Matthew","Neil","Peter","Roger",
                  "Stephen"]
ibd_name_targ2 = ["Aisha","Keisha","Lakisha","Latisha","Latoya","Malika","Nichelle","Shereen","Tamika","Tanisha",
                  "Yolanda","Yvette"]
ibd_name_attr1 = [
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
ibd_name_attr2 = [
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

# IBD_term_word
ibd_term_targ1 = [
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
ibd_term_targ2 = [
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
ibd_term_attr1 = [
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
ibd_term_attr2 = [
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

# EIBD_name_word
eibd_name_targ1 = ["Andrew","Brad","Frank","Geoffrey","Jack","Jonathan","Josh","Matthew","Neil","Peter","Roger",
                  "Stephen"]
eibd_name_targ2 = ["Aisha","Keisha","Lakisha","Latisha","Latoya","Malika","Nichelle","Shereen","Tamika","Tanisha",
                  "Yolanda","Yvette"]
eibd_name_attr1 = ["arrogant","blond","high-status","intelligent","racist","rich","sucessful","tall"]
eibd_name_attr2 = ["aggressive","bigbutt","confident","darkskinned","fried-chicken","overweight","promiscuous",
                   "unfeminine"]

# EIBD_term_word
eibd_term_targ1 = [
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
eibd_term_targ2 = [
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
eibd_term_attr1 = ["arrogant","blond","high-status","intelligent","racist","rich","sucessful","tall"]
eibd_term_attr2 = ["aggressive","bigbutt","confident","darkskinned","fried-chicken","overweight","promiscuous",
                   "unfeminine"]

def shorten_sent(sent, wd):
    """ Function to shorten the raw comment
    take window of size 9 around word of interest for single word stimuli
    take window of size 13 around word of interest for multiple word stimuli
    """
    multiple_words = False
    if len(wd.split()) > 1: # case: multiple word stimuli
        window_size = 13
        multiple_words = True
    else: # case: single word stimuli
        window_size = 9

    wds = sent.split()
    if len(wds) >= window_size:
        if multiple_words:
            # determine idx of stimuli in input sentence
            idx_start = wds.index(wd.split()[0])
            idx_end = idx_start + len(wd.split()) - 1
        else:
            idx_start, idx_end = wds.index(wd), wds.index(wd)

        # case: take first window_size words
        if idx_start < ((window_size-1)/2):
            wds_used = wds[:window_size]
        # case: take last window_size words
        elif (len(wds) - idx_end - 1) < ((window_size-1)/2):
            wds_used = wds[-window_size:]
        # case: take (window_size/2) words before and after stimuli
        else:
            if multiple_words:
                window = math.ceil((window_size - len(wd.split())) / 2)
                wds_used = wds[(idx_start - window):(idx_end + window)]
            else:
                wds_used = wds[int((idx_start-((window_size-1)/2))):int((idx_end+((window_size-1)/2))+1)]
        new_sent = ' '.join(wds_used)
    else:
        new_sent = sent
    return new_sent

def get_stimuli(test_name):
    """ Function to get stimuli for specified bias test """
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
    elif test_name == 'c9m_name':
        targ1, targ2, attr1, attr2 = c9m_name_targ1, c9m_name_targ2, c9m_name_attr1, c9m_name_attr2
    elif test_name == 'c9_term':
        targ1, targ2, attr1, attr2 = c9_term_targ1, c9_term_targ2, c9_term_attr1, c9_term_attr2
    elif test_name == 'dis_term':
        targ1, targ2, attr1, attr2 = dis_term_targ1, dis_term_targ2, dis_term_attr1, dis_term_attr2
    elif test_name == 'dism_term':
        targ1, targ2, attr1, attr2 = dism_term_targ1, dism_term_targ2, dism_term_attr1, dism_term_attr2
    elif test_name == 'occ_name':
        targ1, targ2, attr1, attr2 = occ_name_targ1, occ_name_targ2, occ_name_attr1, occ_name_attr2
    elif test_name == 'occ_term':
        targ1, targ2, attr1, attr2 = occ_term_targ1, occ_term_targ2, occ_term_attr1, occ_term_attr2
    elif test_name == 'ibd_name':
        targ1, targ2, attr1, attr2 = ibd_name_targ1, ibd_name_targ2, ibd_name_attr1, ibd_name_attr2
    elif test_name == 'ibd_term':
        targ1, targ2, attr1, attr2 = ibd_term_targ1, ibd_term_targ2, ibd_term_attr1, ibd_term_attr2
    elif test_name == 'eibd_name':
        targ1, targ2, attr1, attr2 = eibd_name_targ1, eibd_name_targ2, eibd_name_attr1, eibd_name_attr2
    elif test_name == 'eibd_term':
        targ1, targ2, attr1, attr2 = eibd_term_targ1, eibd_term_targ2, eibd_term_attr1, eibd_term_attr2
    else:
        raise ValueError("Stimuli for bias test %s not found!" % test_name)
    return targ1, targ2, attr1, attr2

def create_batches(sent_lst):
      """ Function to generate sent batches """
      if len(sent_lst) != 0:
            batch_size = 10
            n_batches = int((len(sent_lst) - (len(sent_lst) % batch_size)) / batch_size)
            size_batches = [batch_size for _ in range(n_batches)]
            if len(sent_lst) % batch_size != 0: # if applicable add rest (size of last batch < batch_size)
                  size_batches = size_batches + [len(sent_lst) % batch_size]

            sents_batch = []
            # list containing sents per batch
            for i in range(len(size_batches)-1):
                  idx_start = i * size_batches[i]
                  idx_end = idx_start + size_batches[i]
                  sents_batch.append(sent_lst[idx_start:idx_end])
            sents_batch.append(sent_lst[-(size_batches[-1]):])
      else:
            sents_batch = []
      return sents_batch

def load_model(model_name):
    """ Load language model and corresponding tokenizer if applicable """
    if model_name == 'bert':
          model = BertModel.from_pretrained('bert-base-cased')
          tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
          # additional 'Fast' BERT tokenizer for subword tokenization ID mapping
          subword_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    else:
          raise ValueError("Model %s not found!" % model_name)

    return model.to(device), tokenizer, subword_tokenizer

def bert(sent_dict, test_name):
    """ Function to encode sentences with BERT """

    targ1_lst, targ2_lst, attr1_lst, attr2_lst = get_stimuli(test_name)
    wd_list = targ1_lst + targ2_lst + attr1_lst + attr2_lst
    out_dict = {wd:{'sent': [],
                    'word-average': [],
                    'word-start': [],
                    'word-end': []} for wd in wd_list}

    bert_model, bert_tok, bert_sub_tok = load_model('bert')

    print(f'Starting to generate embeddings for bias test {test_name}')
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    for wd in wd_list:
          batches = create_batches(sent_dict[wd])
          for batch in batches:
                batch = [shorten_sent(sent, wd) for sent in batch]
                # [CLS] and [SEP] tokens are added automatically
                encodings = bert_tok(batch, return_tensors='pt', padding=True, truncation=True)
                token_ids = torch.tensor(encodings['input_ids'], device=device)
                # map tokens to input words
                subword_ids = [bert_sub_tok(sent, add_special_tokens=False).word_ids() for sent in batch]
                vecs = bert_model(input_ids=token_ids)

                for idx_sent, sent in enumerate(batch):
                      for encoding, value in out_dict[wd].items():
                            if encoding[:4] == 'word':  # here: subword tokenization
                                  if len(wd.split()) > 1: # case: multiple words
                                        # determine idx of stimuli in input sentence; account for [CLS] token
                                        idx_start = sent.split().index(wd.split()[0]) + 1
                                        # account for [CLS] token; range function excludes end idx
                                        idx_end = idx_start + len(wd.split())
                                        # obtain vecs of all relevant tokens
                                        token_vecs = []
                                        for idxs in range(idx_start, idx_end):
                                              token_vecs.append(
                                                    vecs['last_hidden_state'][idx_sent][idxs].cpu().detach().numpy())
                                        # extract rep of token of interest as average over all tokens
                                        out_dict[wd][encoding].append(np.mean(np.asarray(token_vecs), axis=0))
                                  else:
                                        # determine idx of stimulus in input sentence
                                        idx = sent.split().index(wd)
                                        if '-' in sent.split()[idx]:  # case: special example of subword tokenization
                                              idx_stimuli = [i for i, element in enumerate(subword_ids[idx_sent]) if
                                                             element == idx]
                                              # account for [CLS] token
                                              idx_start = idx_stimuli[0] + 1
                                              idxs_first_part = len(idx_stimuli)
                                              idxs_second_part = len(
                                                    [i for i, element in enumerate(subword_ids[idx_sent]) if
                                                     element == (idx_start + 1)])
                                              # account for [CLS] token; range function excludes end idx
                                              idx_end = idx_start + idxs_first_part + idxs_second_part + 1
                                              if encoding == 'word-average':
                                                    # obtain vecs of all relevant tokens
                                                    token_vecs = []
                                                    for idxs in range(idx_start, idx_end):
                                                          token_vecs.append(vecs['last_hidden_state'][idx_sent][
                                                                                  idxs].cpu().detach().numpy())
                                                    # extract rep of token of interest as average over all tokens
                                                    out_dict[wd][encoding].append(np.mean(np.asarray(token_vecs),
                                                                                          axis=0))
                                              elif encoding == 'word-start':
                                                    out_dict[wd][encoding].append(vecs['last_hidden_state'][idx_sent][
                                                                                   idx_start].cpu().detach().numpy())
                                              elif encoding == 'word-end':
                                                    idx_new = idx_start + idxs_first_part + idxs_second_part
                                                    out_dict[wd][encoding].append(
                                                          vecs['last_hidden_state'][idx_sent][idx_new]
                                                                .cpu().detach().numpy())
                                        else:
                                              if subword_ids[idx_sent].count(idx) == 1:  # case: no subword tokenization
                                                    # account for [CLS] token
                                                    idx_new = idx + 1
                                                    # extract rep of token of interest
                                                    out_dict[wd][encoding].append(
                                                          vecs['last_hidden_state'][idx_sent][idx_new]
                                                                .cpu().detach().numpy())
                                              elif subword_ids[idx_sent].count(idx) > 1:  # case: subword tokenization
                                                    if encoding == 'word-average':
                                                          # obtain vecs of all relevant subwds
                                                          subword_vecs = []
                                                          idx_list = [i for i in range(len(subword_ids[idx_sent])) if
                                                                      subword_ids[idx_sent][i] == idx]
                                                          for idxs in idx_list:
                                                                # account for [CLS] token
                                                                idx_new = idxs + 1
                                                                subword_vecs.append(
                                                                      vecs['last_hidden_state'][idx_sent][idx_new]
                                                                            .cpu().detach().numpy())
                                                          # extract rep of token of interest as average over all subwds
                                                          out_dict[wd][encoding].append(
                                                                np.mean(np.asarray(subword_vecs), axis=0))
                                                    elif encoding == 'word-start':
                                                          # account for CLS token
                                                          idx_new = subword_ids[idx_sent].index(idx) + 1
                                                          # extract rep of token of interest as first subword
                                                          out_dict[wd][encoding].append(
                                                                vecs['last_hidden_state'][idx_sent][idx_new]
                                                                      .cpu().detach().numpy())
                                                    elif encoding == 'word-end':
                                                          # account for [CLS] token
                                                          idx_new = len(subword_ids[idx_sent]) - \
                                                                    subword_ids[idx_sent][::-1].index(idx)
                                                          # extract rep of token of interest as last subword
                                                          out_dict[wd][encoding].append(
                                                                vecs['last_hidden_state'][idx_sent][idx_new]
                                                                      .cpu().detach().numpy())

                            elif encoding == 'sent':
                                  # extract rep of sent as [CLS] token
                                  out_dict[wd][encoding].append(vecs['last_hidden_state'][idx_sent][0]
                                                                .cpu().detach().numpy())

    print(f'Finished generating embeddings for bias test {test_name}')
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    return out_dict

def associate(w,A,B):
    """ Function for
    s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)
    """
    return cosine_similarity(w.reshape(1,-1),A).mean() - cosine_similarity(w.reshape(1,-1),B).mean()

def effect_size(X,Y,A,B):
    """ Function to compute the effect size
    [ mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B) ] /
        [ stddev_{w in X union Y} s(w, A, B) ]
    """
    delta_mean =  np.mean([associate(X[i,:],A,B) for i in range(X.shape[0])]) - \
                  np.mean([associate(Y[i,:],A,B) for i in range(Y.shape[0])])

    XY = np.concatenate((X,Y),axis=0)
    s_union = [associate(XY[i,:],A,B) for i in range(XY.shape[0])]
    std_dev = np.std(s_union,ddof=1)
    var = std_dev**2

    return delta_mean/std_dev, var

def ceat_meta(encs, encoding, N=10000):
    """ Function to run CEAT calculations
    args:
        - encs (Dict[int: Dict])
        - encoding (str): specifies encoding level
        - N (int): number of effect size samples
    """

    weat_dict_targ1 = {wd: encs[0][wd][encoding] for wd in list(encs[0].keys())}
    weat_dict_targ2 = {wd: encs[1][wd][encoding] for wd in list(encs[1].keys())}
    weat_dict_attr1 = {wd: encs[2][wd][encoding] for wd in list(encs[2].keys())}
    weat_dict_attr2 = {wd: encs[3][wd][encoding] for wd in list(encs[3].keys())}

    e_lst = [] # list containing N effect sizes
    v_lst = [] # list containing corresponding N variances

    weat_dicts = [weat_dict_targ1, weat_dict_targ2, weat_dict_attr1, weat_dict_attr2]
    sents_idx_lsts = []
    for weat_dict in weat_dicts:
        sents_dict = {}
        for wd in list(weat_dict.keys()):
            if len(weat_dict[wd]) < N : # case: sample with replacement
                idx_sents = random.choices(range(len(weat_dict[wd])), k=N)
            else: # sample without replacement
                idx_sents = random.sample(range(len(weat_dict[wd])), N)
            sents_dict[wd] = idx_sents
        sents_idx_lsts.append(sents_dict)

    for i in range(N):
        X = np.array([weat_dict_targ1[wd][sents_idx_lsts[0][wd][i]] for wd in list(weat_dict_targ1.keys())])
        Y = np.array([weat_dict_targ2[wd][sents_idx_lsts[1][wd][i]] for wd in list(weat_dict_targ2.keys())])
        A = np.array([weat_dict_attr1[wd][sents_idx_lsts[2][wd][i]] for wd in list(weat_dict_attr1.keys())])
        B = np.array([weat_dict_attr2[wd][sents_idx_lsts[3][wd][i]] for wd in list(weat_dict_attr2.keys())])
        e,v = effect_size(X,Y,A,B)
        e_lst.append(e)
        v_lst.append(v)

    # random-effects model from meta-analysis literature
    e_array = np.array(e_lst)
    w_array = 1 / np.array(v_lst)

    # total variance Q
    q1 = np.sum(w_array*(e_array**2))
    q2 = ((np.sum(e_array*w_array))**2)/np.sum(w_array)
    q = q1 - q2
    df = N - 1 # degrees of freedom

    # variance decomposition:
    # if only source of variance is within-study var then Q = df
    # thus compute between-studies var tao_square (excess variance)
    if q > df:
        # c : scaling factor such that tao_square is in same metric as within-study var
        c = np.sum(w_array) - np.sum(w_array**2)/np.sum(w_array)
        tao_square = (q-df)/c
    else:
        tao_square = 0

    # weighting of each study by the inverse of its variance
    v_star_array = np.array(v_lst) + tao_square # var includes all variance components
    w_star_array = 1/v_star_array

    # combined effect size (weighted mean)
    ces = np.sum(w_star_array*e_array)/np.sum(w_star_array)
    v = 1/np.sum(w_star_array)
    z = ces/np.sqrt(v)
    # 2-tailed p-value, standard normal cdf (by CLS)
    p_value = scipy.stats.norm.sf(z, loc = 0, scale = 1)

    return ces, p_value

sent_dict = pickle.load(open('sent_dict_single.pickle','rb'))

all_tests = ['c1_name', 'c3_name', 'c3_term', 'c6_name', 'c6_term', 'c9_name', 'c9m_name', 'c9_term',
             'occ_name', 'occ_term', 'dis_term', 'dism_term', 'ibd_name', 'ibd_term', 'eibd_name', 'eibd_term']
results = []

for test in all_tests:
      embeds = bert(sent_dict, test)

      targ1, targ2, attr1, attr2 = get_stimuli(test)
      encs = {}
      i = 0
      # map embeddings to respective word set
      for concept in [targ1, targ2, attr1, attr2]:
            encs_concept = {stimulus: embeds[stimulus] for stimulus in concept}
            encs[i] = encs_concept
            i += 1

      # check if there exist reps for all word sets; delete all stimuli with no reps
      # take 'sent' encoding level as representative
      omit_test = False
      for i in range(4):
            # if all stimuli for a word set are missing then omit test in next step (bool)
            if all(len(encs[i][wd]['sent']) == 0 for wd in list(encs[i].keys())):
                  omit_test = True
            # if some stimuli in word set are missing then delete missing stimuli
            elif any(len(encs[i][wd]['sent']) == 0 for wd in list(encs[i].keys())):
                  encs[i] = {wd: encs[i][wd] for wd in list(encs[i].keys()) if len(encs[i][wd]['sent']) != 0}

      if not omit_test:
            # if applicable downsample to smallest target word set
            if len(encs[0].keys()) != len(encs[1].keys()):
                  min_n = min([len(encs[0].keys()),len(encs[1].keys())])
                  # randomly sample min number of stimuli for both word sets
                  if not len(encs[0].keys()) == min_n:
                        wd_lst_new = random.sample(list(encs[0].keys()), min_n)
                        encs[0] = {i: encs[0][i] for i in wd_lst_new}
                  else:
                        wd_lst_new = random.sample(list(encs[1].keys()), min_n)
                        encs[1] = {i: encs[1][i] for i in wd_lst_new}
      else:
            break

      for encoding in ['sent', 'word-average', 'word-start', 'word-end']:
            # default parameter: N = 10,000
            esize, pval = ceat_meta(encs, encoding)
            results.append(dict(
                  method='CEAT',
                  test=test,
                  model='bert',
                  evaluation_measure='cosine',
                  context='reddit',
                  encoding_level=encoding,
                  p_value=pval,
                  effect_size=esize))

# save results and specs of code run (time, date)
results_path = time.strftime("%Y%m%d-%H%M%S") + '_CEAT_bert_reddit.csv'
print('Writing results to {}'.format(results_path))
with open(results_path, 'w') as f:
      writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
      writer.writeheader()
      for r in results:
            writer.writerow(r)
