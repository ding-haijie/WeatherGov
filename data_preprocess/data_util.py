import pickle

import matplotlib.pyplot as plt

"""
1. get distribution of length of the samples' text & aligns
2. build vocabulary
"""

with open('../data/weathergov_data.pkl', 'rb') as f:
    weather_data = pickle.load(f)

max_len_text = -1
average_len_text = 0
max_len_align = -1
average_len_align = 0

text_len = {idx: 0 for idx in range(1, 89)}
align_len = {idx: 0 for idx in range(1, 16)}
align_dict = {idx: 0 for idx in range(36)}

for idx_data in weather_data:
    text = idx_data['text'].split()
    average_len_text += len(text)
    if len(text) > max_len_text:
        max_len_text = len(text)
    average_len_align += len(idx_data['align'])
    if len(idx_data['align']) > max_len_align:
        max_len_align = len(idx_data['align'])
    text_len[len(text)] += 1

    align = idx_data['align']
    align_len[len(align)] += 1
    for each_align in align:
        align_dict[each_align] += 1

print(f'max_len_text_weathergov: {max_len_text}')
print(f'average_len_text_weathergov: {average_len_text / len(weather_data)}')
print(f'max_len_align_weathergov: {max_len_align}')
print(f'average_len_align_weathergov: {average_len_align / len(weather_data)}')


def plot_show(plt_dict, name):
    plt.bar(list(plt_dict.keys()), plt_dict.values())
    plt.xlabel(name)
    plt.ylabel('number')
    plt.title('distribution of ' + name)
    plt.show()


plot_show(text_len, 'text_len_weather_gov')
plot_show(align_len, 'align_len')
plot_show(align_dict, 'align')

# build vocabulary for weathergov from samples' text
word2ind = dict()

word2ind['SOS_TOKEN'] = 0
word2ind['EOS_TOKEN'] = 1
word2ind['PAD_TOKEN'] = 2
word2ind['UNK_TOKEN'] = 3

idx = 4
for sample in weather_data:
    token_list = sample['text'].split()
    for token in token_list:
        if token not in word2ind:
            word2ind[token] = idx
            idx += 1

ind2word = {value: key for key, value in word2ind}

vocabulary = {'word2ind': word2ind,
              'ind2word': ind2word,
              'vocab_size': len(word2ind)}

with open('../data/weathergov_vocab.pkl', 'wb') as f:
    pickle.dump(vocabulary, f)
