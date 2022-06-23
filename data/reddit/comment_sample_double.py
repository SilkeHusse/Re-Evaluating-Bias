""" sample comments for double word dataset """
import pickle
import random

random.seed(1111)
N = 10000

def merge_dict(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
           dict_3_value = [value , dict_1[key]]
           dict_3[key] = [item for sublist in dict_3_value for item in sublist]
   return dict_3

# TODO set parameter, default = False
simplified = False

# merge sent_dict_double files; starting point is dataset for month 1
if simplified:
    dataset = pickle.load(open('processed_data/sent_dict_double_simplified_1.pickle', 'rb'))
else:
    dataset = pickle.load(open('processed_data/sent_dict_double_1.pickle','rb'))

for i in range(2,13):
    if simplified:
        dataset_to_merge_name = 'processed_data/sent_dict_double_simplified_' + str(i) + '.pickle'
    else:
        dataset_to_merge_name = 'processed_data/sent_dict_double_' + str(i) + '.pickle'
    dataset_to_merge = pickle.load(open(dataset_to_merge_name,'rb'))
    for key, value in dataset.items():
        dataset[key] = merge_dict(dataset[key], dataset_to_merge[key])
    print(f'sent_dict_double: Merged month {i}')

# if applicable, sample N = 10,000 comments
for key, value in dataset.items():
    for key_tuple, value_tuple in dataset[key].items():
        if len(dataset[key][key_tuple]) > N:
            dataset[key][key_tuple] = random.sample(dataset[key][key_tuple], N)

if simplified:
    pickle.dump(dataset, open('sent_dict_double_simplified.pickle', 'wb'))
else:
    pickle.dump(dataset, open('sent_dict_double.pickle', 'wb'))
