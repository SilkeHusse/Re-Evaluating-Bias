""" Function to replace stimuli in template sentences
and thus generate sentences for SEAT """

def replace(test, dataset, template_sents, form='normal'):

    stimuli = {'targ1': None, 'targ2': None, 'attr1': None, 'attr2': None}
    sents = {'targ1': [], 'targ2': [], 'attr1': [], 'attr2': []}
    test = test.lower()[:-5]

    multiple_targ = False  # bool indicating if targ stimuli consist of multiple words
    multiple_attr = False  # bool indicating if attr stimuli consist of multiple words

    if form == 'reduced':
        if test == 'c1_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c3_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c6_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c6_term':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c9_name' or test == 'c9_name_m' or test == 'c9_term':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'occ_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
        elif test == 'occ_term':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
        else:
            raise ValueError("Reduced dataset of bias test %s not found!" % test)
    elif form == 'simplified':
        if test == 'c1_name':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['targ']['plural_thing']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c3_term':
            multiple_targ = True
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', 'a ' + stimulus + ' person'))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', 'a ' + stimulus + ' person'))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus + ' people'))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus + ' people'))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c6_term':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c9_name' or test == 'c9_name_m':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', 'a ' + stimulus + ' disease'))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', 'a ' + stimulus + ' disease'))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'occ_term':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
        else:
            raise ValueError("Simplified dataset of bias test %s not found!" % test)
    else:
        if test == 'c1_name':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['targ']['plural_thing']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c3_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c3_term':
            multiple_targ = True
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c6_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c6_term':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'c9_name' or test == 'c9_name_m' or test == 'c9_term':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'occ_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
        elif test == 'occ_term':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', 'a ' + stimulus))
        elif test == 'dis_term' or test == 'dis_term_m':
            multiple_targ = True
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'i1_name' or test == 'i2_name':
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        elif test == 'i1_term' or test == 'i2_term':
            multiple_targ = True
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'].append(sent.replace('AAA', stimulus))
        else:
            raise ValueError("Bias test %s not found!" % test)

    return stimuli, sents, multiple_targ, multiple_attr
