import json
import os
from collections import OrderedDict

from src.recommendationlab import config


class Vocab:
    def __init__(self, data):
        self.data = data
        self._init_data()
        self.id2item = {self.item2id[k]: k for k in self.item2id}

    def _init_data(self):
        self.item2id = OrderedDict()
        self.item2id['UNK'] = 0
        for d in self.data:
            if d == 'UNK':
                continue
            self.item2id[d] = len(self.item2id)

    @property
    def items(self):
        return self.item2id.keys()

    @staticmethod
    def load_vocab(vocab_file):
        with open(os.path.join(config.SPLITSPATH, vocab_file), 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        return vocab
