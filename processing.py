# load all annotated files (json format)
# load and convert each dataset separately, then merge and shuffle

import json
import random
import spacy
from spacy.tokens import DocBin
from spacy.cli.train import train

pilot2_file_name = 'pm_pilot2_sample_ann_lk.json'
lk_file_name = 'pm_sample_lk_ann.json'
ss_file_name = 'pm_sample_ss_ann.json'
fv_file_name = 'pm_sample_fv_ann.json'
sw_file_name = 'pm_sample_sw_ann.json'

def convert_json(file_name):
    '''
    takes string (file_name) - file name of the json file to be read
    returns data in the file in spacy friendly format
    '''

    f = open(file_name, encoding='UTF-8')
    data = json.load(f)
    new_list = []

    for annotation in data:
        ents = []
        for ent in annotation['annotations'][0]['result']:
            ents.append((ent['value']['start'], ent['value']['end'], ent['value']['labels'][0]))
        new_tuple = (annotation['data']['affiliation'], ents)
        #ents = [tuple(entity[:3]) for entity in annotation['entities']]
        new_list.append(new_tuple)

    return new_list

# load and convert each dataset, combine in annotated_data
annotated_data = []

annotated_data.extend(convert_json(pilot2_file_name))
print("loaded pilot 2")
annotated_data.extend(convert_json(lk_file_name))
print("loaded lk")
annotated_data.extend(convert_json(sw_file_name))
print("loaded sw")
annotated_data.extend(convert_json(ss_file_name))
print("loaded ss")
annotated_data.extend(convert_json(fv_file_name))
print("loaded fv")

print(f"Annotated data length: {len(annotated_data)}")

# shuffle the data
random.seed(30)
random.shuffle(annotated_data)

# calculate training, valid, and test sizes
train_size = int(len(annotated_data) * 0.7)
valid_size = int(len(annotated_data) * 0.1)
test_size = len(annotated_data) - train_size - valid_size

# split the tuple list into training, valid, and test sets
training_data = annotated_data[:train_size]
valid_data = annotated_data[train_size:train_size+valid_size]
test_data = annotated_data[train_size+valid_size:]

# create blank spacy model
nlp = spacy.blank('en')

# convert the datasets to spacy binary format
train_db = DocBin ()
success_count = 0
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label = label)
        ents.append(span)
    try:
        doc.ents = ents
        success_count += 1
    except:
        print(f"Couldn't convert {text} to binary")
    train_db.add(doc)
print(train_db)
print(len(train_db))
print(f"Successfully converted {success_count} annotations to binary for train_db")
train_db.to_disk("./train.spacy")

valid_db = DocBin ()
success_count = 0
for text, annotations in valid_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label = label)
        ents.append(span)
    try:
        doc.ents = ents
        success_count += 1
    except:
        print(f"Couldn't convert {text} to binary")
    valid_db.add(doc)
print(valid_db)
print(len(valid_db))
print(f"Successfully converted {success_count} annotations to binary for valid_db")
valid_db.to_disk("./valid.spacy")

test_db = DocBin ()
success_count = 0
for text, annotations in test_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label = label)
        ents.append(span)
    try:
        doc.ents = ents
        success_count += 1
    except:
        print(f"Couldn't convert {text} to binary")
    test_db.add(doc)
print(test_db)
print(len(test_db))
print(f"Successfully converted {success_count} annotations to binary for test_db")
test_db.to_disk("./test.spacy")


