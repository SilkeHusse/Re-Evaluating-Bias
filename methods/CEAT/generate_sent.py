""" Function to replace stimuli in template sentences 
and thus generate sentences for CEAT """

def replace(test, dataset, template_sents, form='normal'):
    
    stimuli = {'targ1': None, 'targ2': None, 'attr1': None, 'attr2': None}
    test = test.lower()[:-5]
    
    if form == 'reduced':
        # define lists of bias tests sharing same specifications
        specs_sp = ['c1_name', 'c3_name', 'c6_name', 'cc6_term']
        specs_s = ['occ_name', 'occ_term', 'c9_term', 'c9_name', 'c9_name_m']

        if test in specs_sp:
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
        elif test in specs_s:
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
        else:
            raise ValueError("Reduced dataset of bias test %s not found!" % test)
        
    elif form == 'simplified':
        # define lists of bias tests sharing same specifications
        specs_sp = ['c1_name', 'c3_term', 'c6_term']
        specs_s = ['c9_name', 'c9_name_m']

        if test in specs_sp:
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
        elif test in specs_s:
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
        elif test == 'occ_term':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
        else:
            raise ValueError("Simplified dataset of bias test %s not found!" % test)
        
    else:
        # define lists of bias tests sharing same specifications
        specs_sp = ['c1_name', 'c3_term', 'c6_term']
        specs_s = ['i1_term', 'i2_term', 'i1_name', 'i2_name', 'dis_term', 'dis_term_m', 'occ_name', 'c9_term',
                   'c9_name', 'c9_name_m']

        if test in specs_sp:
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
        elif test in specs_s:
            stimuli['targ1'] = dataset['targ1']['singular']
            stimuli['targ2'] = dataset['targ2']['singular']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
        elif test == 'c3_name' or test == 'c6_name':
            stimuli['targ1'] = dataset['targ1']['singular'] 
            stimuli['targ2'] = dataset['targ2']['singular'] 
            stimuli['attr1'] = dataset['attr1']['singular'] + dataset['attr1']['plural']
            stimuli['attr2'] = dataset['attr2']['singular'] + dataset['attr2']['plural']
        elif test == 'occ_term':
            stimuli['targ1'] = dataset['targ1']['singular'] + dataset['targ1']['plural']
            stimuli['targ2'] = dataset['targ2']['singular'] + dataset['targ2']['plural']
            stimuli['attr1'] = dataset['attr1']['singular']
            stimuli['attr2'] = dataset['attr2']['singular']
        else:
            raise ValueError("Bias test %s not found!" % test)

    sents = {'targ1': {i: [] for i in stimuli['targ1']}, 'targ2': {i: [] for i in stimuli['targ2']},
             'attr1': {i: [] for i in stimuli['attr1']}, 'attr2': {i: [] for i in stimuli['attr2']}}

    # create sents by replacing target and attribute words in template sentences
    if form == 'reduced':
        if test == 'c1_name':
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c3_name':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c6_name':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c6_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c9_name' or test == 'c9_name_m' or test == 'c9_term':
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'occ_name':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
        elif test == 'occ_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
    elif form == 'simplified':
        if test == 'c1_name':
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['targ']['plural_thing']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c3_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' person'))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' person'))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus + ' people'))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus + ' people'))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c6_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c9_name' or test == 'c9_name_m':
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' disease'))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', 'a ' + stimulus + ' disease'))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'occ_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))

    else:
        if test == 'c1_name':
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['targ']['plural_thing']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c3_name':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c3_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c6_name':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c6_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
            for sent in template_sents['attr']['plural_basic']:
                for stimulus in dataset['attr1']['plural']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['plural']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'c9_name' or test == 'c9_name_m' or test == 'c9_term':
            for sent in template_sents['targ']['singular_thing']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'occ_name':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
        elif test == 'occ_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['targ']['plural_person']:
                for stimulus in dataset['targ1']['plural']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['plural']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_basic']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', 'a ' + stimulus))
        elif test == 'dis_term' or test == 'dis_term_m':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'i1_name' or test == 'i2_name':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
        elif test == 'i1_term' or test == 'i2_term':
            for sent in template_sents['targ']['singular_person']:
                for stimulus in dataset['targ1']['singular']:
                    sents['targ1'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
                for stimulus in dataset['targ2']['singular']:
                    sents['targ2'][stimulus].append(sent.replace('TTT', 'a ' + stimulus))
            for sent in template_sents['attr']['singular_time']:
                for stimulus in dataset['attr1']['singular']:
                    sents['attr1'][stimulus].append(sent.replace('AAA', stimulus))
                for stimulus in dataset['attr2']['singular']:
                    sents['attr2'][stimulus].append(sent.replace('AAA', stimulus))
    
    return stimuli, sents