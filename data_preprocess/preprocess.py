import os
import pickle
import random
import re

import mysql.connector as mysql_connector

"""
dataset: weathergov[https://cs.stanford.edu/~pliang/data/weather-data.zip]
take all separated files into one integrated pickle file
"""

conn = mysql_connector.connect(host="[host]", port="[port]",
                               user="[username]", password="[password]",
                               database="[db_name]")
cursor = conn.cursor()

path = '../data/weathergov/data'
states_all = os.listdir(path)
states_all.sort()

samples_data_list = []
persist_list = []

for state_name in states_all:
    # iterate states folder
    cities_all = os.listdir(os.path.abspath(path) + '/' + state_name)
    cities_all.sort()

    for city_name in cities_all:
        # iterate cities folder
        single_file_list = os.listdir(os.path.abspath(path) + '/' + state_name + '/' + city_name)
        single_file_list.sort()

        # samples_list contains the three related files (i.e. ".align", ".events", ".text" which belong to one sample)
        samples_list = [single_file_list[i:i + 3] for i in range(0, len(single_file_list), 3)]
        for per_sample in samples_list:
            # iterate samples folder
            data_dict = {}
            persist_dict = {}
            for per_sample_path in per_sample:
                if '.align' in per_sample_path:
                    with open(
                            os.path.abspath(path) + '/' + state_name + '/' + city_name + '/' + per_sample_path,
                            encoding='utf-8') as f:
                        line = f.readline()
                        list_tmp = []
                        while line:
                            # each line contains a set of (line number of the text file,
                            # line number of the events file to which the line of text is aligned)
                            for value in line.split()[1:]:
                                list_tmp.append(int(value))
                            line = f.readline()
                    data_dict['align'] = list_tmp
                    persist_dict['align'] = str(list_tmp)
                elif '.events' in per_sample_path:
                    with open(
                            os.path.abspath(path) + '/' + state_name + '/' + city_name + '/' + per_sample_path,
                            encoding='utf-8') as f:
                        events = ""
                        line = f.readline()  # each id_number's relevant data per line
                        idx = 0
                        while line:
                            events += line + "\n"
                            list_tmp = line.split()
                            dict_idx = {}
                            for text in list_tmp:
                                if '.type' in text:
                                    dict_idx['type'] = text[text.index(':') + 1:]
                                if '.label' in text:
                                    dict_idx['label'] = text[text.index(':') + 1:]
                                if '@time' in text:
                                    dict_idx['time'] = text[text.index(':') + 1:]
                                if '#min' in text:
                                    dict_idx['min'] = text[text.index(':') + 1:]
                                if '#mean' in text:
                                    dict_idx['mean'] = text[text.index(':') + 1:]
                                if '#max' in text:
                                    dict_idx['max'] = text[text.index(':') + 1:]
                                if '@mode' in text:
                                    dict_idx['mode'] = text[text.index(':') + 1:]
                                if '@mode-bucket-0-20-2' in text:
                                    dict_idx['mode_bucket_0_20_2'] = text[text.index(':') + 1:]
                                if '@mode-bucket-0-100-4' in text:
                                    dict_idx['mode_bucket_0_100_4'] = text[text.index(':') + 1:]
                            if 'mode_bucket_0_20_2' not in dict_idx:
                                dict_idx['mode_bucket_0_20_2'] = ''
                            data_dict['id' + str(idx)] = dict_idx
                            idx += 1
                            line = f.readline()
                        persist_dict['events'] = str(events)
                elif '.text' in per_sample_path:
                    with open(
                            os.path.abspath(path) + '/' + state_name + '/' + city_name + '/' + per_sample_path,
                            encoding='utf-8') as f:
                        line = f.readline()
                        list_tmp = []
                        while line:
                            list_tmp.append(re.sub('\n', ' ', line).lower())
                            line = f.readline()
                        data_dict['text'] = ''.join(list_tmp)
                        persist_dict['text'] = ''.join(list_tmp)
                else:
                    pass
                date = per_sample_path.split('.')[0]
                data_dict['date'] = date
                data_dict['city'] = city_name
                data_dict['state'] = state_name
                offset = date.split("-")[-1]
                date = "-".join(date.split("-")[:-1])
                persist_dict['date'] = date
                persist_dict['offset'] = int(offset)
                persist_dict['state'] = state_name
                persist_dict['city'] = city_name
            samples_data_list.append(data_dict)
            persist_list.append(persist_dict)

# randomly shuffle the dataset
random.shuffle(samples_data_list)

# persist into pickle file
with open('../data/weathergov_data.pkl', 'wb') as f:
    pickle.dump(samples_data_list, f)

# persist into mysql database
for sample in persist_list:
    cursor.execute("insert into weather (events, align, text, city, state, date, offset) values "
                   "(%s, %s, %s, %s, %s, %s, %s)",
                   (sample['events'], sample['align'], sample['text'],
                    sample['city'], sample['state'], sample['date'], sample['offset']))
conn.commit()
cursor.close()
conn.close()
