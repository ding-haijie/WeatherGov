import pickle
import numpy as np

from data_preprocess.get_info import get_info
from config import params


class DataProcessor(object):

    def __init__(self):
        super(DataProcessor, self).__init__()
        self.num_info = params['num_info']
        self.dim_info = params['input_dim']
        self.max_len = params['max_len']

        with open('./data/weathergov_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        self.ind2word = vocab['ind2word']
        self.word2ind = vocab['word2ind']
        del vocab
        self.SOS_TOKEN = self.word2ind['SOS_TOKEN']
        self.EOS_TOKEN = self.word2ind['EOS_TOKEN']
        self.PAD_TOKEN = self.word2ind['PAD_TOKEN']
        self.UNK_TOKEN = self.word2ind['UNK_TOKEN']

        with open('./data/weathergov_data.pkl', 'rb') as f:
            samples = pickle.load(f)
        self.test_data = samples[26764:]  # 2764
        del samples

    def process_seq(self, tag="train"):
        with open('./data/weathergov_data.pkl', 'rb') as f:
            samples = pickle.load(f)
        if tag == 'train':
            data = samples[:24000]  # 24000
        elif tag == 'dev':
            data = samples[24000: 26764]  # 2764
        else:
            raise ValueError('invalid data_tag: ', tag)
        del samples

        seq_info_numpy = np.zeros((len(data), self.num_info, self.dim_info))
        seq_target_numpy = np.full((len(data), self.max_len, 1), self.PAD_TOKEN, dtype=np.float)

        for data_index, data_item in enumerate(data):
            seq_info_numpy[data_index] = get_info(data_item)
            tokens_text = data_item['text'].split()
            seq_target_numpy[data_index, 0] = self.SOS_TOKEN
            idx_pos = 1
            for token in tokens_text:
                if token in self.word2ind:
                    seq_target_numpy[data_index, idx_pos] = self.word2ind[token]
                else:
                    seq_target_numpy[data_index, idx_pos] = self.UNK_TOKEN
                idx_pos += 1

        return seq_info_numpy, seq_target_numpy

    def process_one_data(self, idx_data=0):
        data_item = self.test_data[idx_data]
        seq_info_numpy = get_info(data_item)

        return seq_info_numpy

    def get_refs(self):
        list_refs = []
        for data_item in self.test_data:
            list_refs.append(data_item['text'])

        return list_refs

    def get_golds(self):
        list_gold = []
        for data_item in self.test_data:
            list_gold.append(list(set(data_item['align'])))  # remove repetitive aligns

        return list_gold

    def translate(self, list_seq):
        list_token = []
        for index in list_seq:
            if index == self.SOS_TOKEN:
                continue
            elif index == self.PAD_TOKEN:
                continue
            elif index == self.UNK_TOKEN:
                continue
            elif index == self.EOS_TOKEN:
                break
            else:
                list_token.append(self.ind2word[index])

        return ' '.join(list_token)
