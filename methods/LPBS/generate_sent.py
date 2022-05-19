""" Function to replace stimuli in template sentences
and thus generate sentences for LPBS """

def replace(test, dataset, template_sents, form='normal'):
    
    final_template = {}
    test = test.lower()[:-5]

    if form == 'reduced':

        if test == 'c1_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c3_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c6_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_hobby']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'the AAA'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c6_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_hobby']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'the AAA').replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test[:2] == 'c9':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'occ_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'a AAA'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'occ_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'a AAA').replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        else:
            raise ValueError("Reduced dataset of bias test %s not found!" % test)
    
    elif form == 'simplified':

        if test == 'c1_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
            for sent in template_sents['templates']['plural_basic']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['plural'],
                    'stimuli_targ2': dataset['targ2']['plural'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c3_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_person']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
            for sent in template_sents['templates']['plural_person']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['plural'],
                    'stimuli_targ2': dataset['targ2']['plural'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c6_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_hobby']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'the AAA'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
            for sent in template_sents['templates']['plural_hobby']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['plural'],
                    'stimuli_targ2': dataset['targ2']['plural'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test[:2] == 'c9':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_disease']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'occ_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'a AAA'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
            for sent in template_sents['templates']['plural_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'a AAA'),
                    'stimuli_targ1': dataset['targ1']['plural'],
                    'stimuli_targ2': dataset['targ2']['plural'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        else:
            raise ValueError("Simplified dataset of bias test %s not found!" % test)
        
    else:

        if test == 'c1_name' or test == 'c3_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
            for sent in template_sents['templates']['plural_basic']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['plural'],
                    'stimuli_targ2': dataset['targ2']['plural'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c3_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['adjective'],
                    #'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['adjective']}
                    #'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c6_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_hobby']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'the AAA'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'c6_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_hobby']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'the AAA').replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
            for sent in template_sents['templates']['plural_hobby']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['plural'],
                    'stimuli_targ2': dataset['targ2']['plural'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test[:2] == 'c9' or test[:3] == 'dis':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'occ_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'a AAA'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'occ_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'a AAA').replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
            for sent in template_sents['templates']['plural_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('AAA', 'a AAA'),
                    'stimuli_targ1': dataset['targ1']['plural'],
                    'stimuli_targ2': dataset['targ2']['plural'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'i1_name' or test == 'i2_name':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent,
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        elif test == 'i1_term' or test == 'i2_term':
            idx_template_sent = 0
            for sent in template_sents['templates']['singular_basic']:
                final_template[idx_template_sent] = {
                    'template': sent.replace('TTT', 'the TTT'),
                    'stimuli_targ1': dataset['targ1']['singular'],
                    'stimuli_targ2': dataset['targ2']['singular'],
                    'stimuli_attr1': dataset['attr1']['singular'],
                    'stimuli_attr2': dataset['attr2']['singular']}
                idx_template_sent += 1
        else:
            raise ValueError("Bias test %s not found!" % test)
    
    return final_template