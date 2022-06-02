#test annotator agreement for NER on PubMed Affiliation Fields
import json
from sklearn.metrics import cohen_kappa_score
import pandas as pd


#process json files for each annotator
lk_file_name = 'pm_pilot2_sample_ann_lk.json'
ss_file_name = 'pm_pilot2_sample_ann_ss.json'
fv_file_name = 'pm_pilot2_sample_ann_fv.json'
sw_file_name = 'pm_pilot2_sample_ann_sw.json'

def read_json(file_name):
    '''
    takes string (file_name) - file name of the json file to be read
    returns list of file contents
    '''
    f = open(file_name)
    data = json.load(f)
    new_list = []
    for i in data:
        new_dict = {}
        new_dict['annotations'] = []
        #print(i['annotations'][0]['result'][0]['value']) 
        new_dict['pmid'] = i['data']['pmid']
        for annotation in i['annotations'][0]['result']:
                new_dict['annotations'].append(annotation['value'])
        #print(new_dict)
        #print('\n')
        new_list.append(new_dict)
    f.close()
    #print(new_list)
    return new_list

def order_anns(anns):
    '''
    takes list of dictionaries with 'pmid' fields (anns)
    returns list reordered by pmid
    '''
    ordered_list = sorted(anns, key=lambda d: d['pmid'])
    return(ordered_list)

def dictlist_to_stringlist(dictlist):
    '''
    takes list of dictionaries with 'start' fields (dictlist)
    returns list of dictionaries converted to strings ordered by start values, with 'text' fields taken out
    '''
    string_list = []
    for ann in dictlist:
        new_ann = []
        for entity in ann:
            new_dict = {k: entity[k] for k in entity.keys() - {'text'}}
            new_ann.append(new_dict)
        string_list.append(str(sorted(new_ann, key=lambda d: d['start'])))
    return(string_list)

def dictlist_to_list(dictlist):
    '''
    takes list of dictionaries with 'annotations' fields (dictlist)
    returns list of annotation fields only
    '''
    new_list = []
    for unit in dictlist:
        unit_list = []
        for annotation in unit['annotations']:
            unit_list.append(annotation)
        new_list.append(unit_list)
    return new_list

def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        #print(an1, an2)
        #print('\n')
        if an1 == an2:
            count += 1
            #print(an1)
            #print(an2)
            #print(count)
    A = count / len(ann1)  # observed agreement A (Po)
    #print(count)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)

def agreement_column(ann1,ann_list):
    '''
    take list of annotations for one annotator and list of lists of annotations from all annotators (in order)
    calculate kappa score for agreement between ann1 and everyone else in list
    return list of kappa scores
    '''
    scores = []
    for annotator in ann_list:
        scores.append(cohen_kappa(ann1,annotator))
    return(scores)

def log_disagreements(ann1,ann2):
    '''
    take two lists of annotations (ann1, ann2)
    return dataframe of disagreements
    '''
    #new_dict = {'ann1':[],'ann2':[],'num_entities_dif':[],'num_categories_dif':[],'num_mismatched_lengths':[]}
    new_dict = {'ann1':[],'ann2':[],'num_entities_dif':[],'num_categories_dif':[],'num_org_dif':[],'num_state_dif':[],'num_country_dif':[],'num_mismatched_lengths':[]}
    for an1, an2 in zip(ann1, ann2):
        a1 = an1['annotations']
        a2 = an2['annotations']
        a1_text = dictlist_to_stringlist([a1])
        a2_text = dictlist_to_stringlist([a2])
        if a1_text[0] != a2_text[0]:
            new_dict['ann1'].append(a1)
            new_dict['ann2'].append(a2)
            new_dict['num_entities_dif'].append(len(a1)-len(a2))
            org1_count = 0
            state1_count = 0
            country1_count = 0
            org2_count = 0
            state2_count = 0
            country2_count = 0
            entity_lengths1 = []
            entity_lengths2 = []
            for e1 in a1:
                if 'ORG' in e1['labels']:
                    org1_count += 1
                if 'STATE' in e1['labels']:
                    state1_count += 1
                if 'COUNTRY' in e1['labels']:
                    country1_count += 1
                entity_lengths1.append(e1['end']-e1['start'])
            for e2 in a2:
                if 'ORG' in e2['labels']:
                    org2_count += 1
                if 'STATE' in e2['labels']:
                    state2_count += 1
                if 'COUNTRY' in e2['labels']:
                    country2_count += 1
                entity_lengths2.append(e2['end']-e2['start'])
            cat1_count = 0
            cat2_count = 0
            for cat in [org1_count, state1_count, country1_count]:
                if cat != 0:
                    cat1_count += 1
            for cat in [org2_count, state2_count, country2_count]:
                if cat != 0:
                    cat2_count += 1
            new_dict['num_categories_dif'].append(cat1_count-cat2_count)
            new_dict['num_org_dif'].append(org1_count-org2_count)
            new_dict['num_state_dif'].append(state1_count-state2_count)
            new_dict['num_country_dif'].append(country1_count-country2_count)
            mismatched_lengths_count = 0
            #print(org1_count, org2_count, state1_count, state2_count, country1_count, country2_count)
            if org1_count == org2_count and state1_count == state2_count and country1_count == country2_count:
                #print('match')
                for length1,length2 in zip(entity_lengths1,entity_lengths2):
                    if length1 != length2:
                        mismatched_lengths_count += 1
                new_dict['num_mismatched_lengths'].append(mismatched_lengths_count)
            else:
                #print('no match')
                new_dict['num_mismatched_lengths'].append("categories not matching")      
            
    new_df = pd.DataFrame(new_dict)
    return new_df

lk_anns_dict = read_json(lk_file_name)
ss_anns_dict = read_json(ss_file_name)
fv_anns_dict = read_json(fv_file_name)
sw_anns_dict = read_json(sw_file_name)


'''
print(lk_anns)
print('\n')
print(ss_anns)
print('\n')
'''

lk_anns_dict = order_anns(lk_anns_dict)
ss_anns_dict = order_anns(ss_anns_dict)
fv_anns_dict = order_anns(fv_anns_dict)
sw_anns_dict = order_anns(sw_anns_dict)

lk_anns = dictlist_to_list(lk_anns_dict)
ss_anns = dictlist_to_list(ss_anns_dict)
fv_anns = dictlist_to_list(fv_anns_dict)
sw_anns = dictlist_to_list(sw_anns_dict)

lk_anns = dictlist_to_stringlist(lk_anns)
ss_anns = dictlist_to_stringlist(ss_anns)
fv_anns = dictlist_to_stringlist(fv_anns)
sw_anns = dictlist_to_stringlist(sw_anns)


'''
print(lk_anns)
print('\n')
print(ss_anns)
'''
#print(cohen_kappa_score(lk_anns, ss_anns))
#print(cohen_kappa(sw_anns,fv_anns))
ann_list = [lk_anns, ss_anns, fv_anns, sw_anns]

agreement_dict = {'annotator':['lk','ss','fv','sw'],'lk':agreement_column(lk_anns,ann_list),'ss':agreement_column(ss_anns,ann_list),'fv':agreement_column(fv_anns,ann_list),'sw':agreement_column(sw_anns,ann_list)}

agreement_df = pd.DataFrame(agreement_dict)
agreement_df.to_csv('pilot_agreement.csv')

lk_ss_dis = log_disagreements(lk_anns_dict, ss_anns_dict)
lk_fv_dis = log_disagreements(lk_anns_dict, fv_anns_dict)
lk_sw_dis = log_disagreements(lk_anns_dict, sw_anns_dict)
ss_fv_dis = log_disagreements(ss_anns_dict, fv_anns_dict)
ss_sw_dis = log_disagreements(ss_anns_dict, sw_anns_dict)
fv_sw_dis = log_disagreements(fv_anns_dict, sw_anns_dict)

lk_ss_dis.to_csv('lk_ss_dis_pilot.csv')
lk_fv_dis.to_csv('lk_fv_dis_pilot.csv')
lk_sw_dis.to_csv('lk_sw_dis_pilot.csv')
ss_fv_dis.to_csv('ss_fv_dis_pilot.csv')
ss_sw_dis.to_csv('ss_sw_dis_pilot.csv')
fv_sw_dis.to_csv('fv_sw_dis_pilot.csv')
