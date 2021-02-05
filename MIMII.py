import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import librosa
import pandas as pd
from tqdm import tqdm
import itertools
from sklearn.preprocessing import LabelEncoder
from scipy.io.wavfile import read as wavread
from sklearn.model_selection import train_test_split
# use with:
# mimii = MIMII(256)
# dataloader = DataLoader(mimii, batch_size=1, pin_memory=True, num_workers=0)
# x, y = next(iter(dataloader))

class MIMII(Dataset):
    all_ids = [0, 2 , 4, 6]
    all_snrs = [6, 0, -6]
    all_types = ['valve', 'pump', 'fan', 'slider']
    def __init__(self, path = 'E:\\FastDatasets\\MIMII\\', length_s=None, channels=None, ids=None, types=None, snrs=None, labels=None, target=None):
        super().__init__()
        if types is None:
            types = ['valve', 'pump', 'fan', 'slider']
        if ids is None:
            ids = [0, 2, 4, 6]
        if snrs is None:
            snrs = [-6, 0, 6]
        if labels is None:
            labels = ['normal', 'abnormal']
        if channels is None:
            channels = [0, 1, 2, 3, 4, 5, 6, 7]
        if target is None:
            target = ['type', 'id']
        if length_s is None:
            length_s = 10
        #assert purpose in ['train', 'val', 'test']
        assert max(channels) < 8
        assert all([type in ['valve', 'pump', 'fan', 'slider'] for type in types])
        assert all([id in [0, 2, 4, 6] for id in ids])
        assert all([snr in [-6, 0, 6] for snr in snrs])
        assert all([t in ['snr', 'type', 'id', 'label'] for t in target])
        assert length_s <= 10.0
        
        self.path = path
        self.samplerate = 16000
        self.ids = ids
        self.types = types
        self.snrs = snrs
        self.channels = channels
        self.labels = labels
        self.data = self.get_data()
        self.target = target
        self.length_s = length_s
        self.wav_length = length_s * self.samplerate
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(self.get_classes())
        self.total_samples = self.data.shape[0]

    def __len__(self):
        return int(self.total_samples * (10 // self.length_s))

    def get_direction_from_type(self, type):
        switcher = {'valve':0, 'pump':2, 'fan':4, 'slider':6}
        if len(type[0]) is not 1: # is list
            return [switcher.get(l) for l in type]
        else:
            return switcher.get(type)

    def get_classes(self):
        in_labels = [self.data[t].unique() for t in self.target]
        classes = list(itertools.product(*in_labels))
        classes = np.array(classes, dtype=np.str)
        classes = np.sum(np.array(classes, dtype=np.object), axis=1)
        return classes

    def one_hot_encode(self, label):
        return np.eye(len(self.get_classes()))[self.labelencoder.transform(label)]

    def one_hot_decode(self, one_hot_label):
        return self.labelencoder.inverse_transform(np.argmax(one_hot_label, axis=-1))

    def get_data(self):
        dict = {}
        i = 0
        #df = pd.DataFrame({'snr':[], 'type':[], 'id':[], 'label':[], 'filepath':[]})
        for snr in self.snrs:
            for type in self.types:
                for id in self.ids:
                    for label in self.labels:
                        subset_path = self.path + str(snr) + '_dB_' + type + '\\' + type + '\\id_0' + str(id) + '\\' + label + '\\'
                        filepaths = [os.path.join(subset_path, f) for f in os.listdir(subset_path) if f.endswith('.wav')]
                        for filepath in filepaths:
                            dict[i] = {'snr':snr, 'type':type, 'id':id, 'label':label, 'filepath':filepath}
                            i = i+1
        df = pd.DataFrame.from_dict(dict, orient="index")
        df = df.sort_values('filepath')
        return df

    @staticmethod
    def get_train_val_test_filepaths(test_ratio=0.2, val_ratio=0.1, shuffle=False, roll=False):

        mimii = MIMII(labels=['normal'], target=['type', 'id', 'snr'])

        if roll is True:
            assert shuffle is False

        if shuffle:
            idxs = np.array([mimii.get_target(idx) for idx in range(mimii.total_samples)])
            idx_train, idx_test, y1, y2 = train_test_split(np.arange(len(idxs)), idxs, test_size=test_ratio, stratify=idxs)
            idx_train, idx_val, y1, y3 = train_test_split(idx_train, y1, test_size=val_ratio, stratify=y1)

        if roll:
            idxs = np.array([mimii.get_target(idx) for idx in range(mimii.total_samples)])
            indexes = np.arange(len(idxs))
            idxs_per_class = [indexes[idxs == cl] for cl in mimii.get_classes()]
            idx_train, idx_val, idx_test = [], [], []
            for idx_cl in idxs_per_class:
                rand_roll = np.random.randint(0, len(idx_cl))
                idx_cl = np.roll(idx_cl, rand_roll)
                indexes = np.roll(indexes, rand_roll)

                idx_train.extend(idx_cl[:int((1-val_ratio) * (1-test_ratio) * len(idx_cl))])
                idx_val.extend(idx_cl[int((1-val_ratio) * (1-test_ratio) * len(idx_cl)):int((1-test_ratio)* len(idx_cl))])
                idx_test.extend(idx_cl[int((1-test_ratio) * len(idx_cl)):])
        else:
            if shuffle == False:
                idxs = np.array([mimii.get_target(idx) for idx in range(mimii.total_samples)])
                indexes = np.arange(len(idxs))
                idxs_per_class = [indexes[idxs==cl] for cl in mimii.get_classes()]
                idx_train, idx_val, idx_test = [], [], []
                for idx_cl in idxs_per_class:
                    idx_train.extend(idx_cl[:int((1 - val_ratio) * (1 - test_ratio) * len(idx_cl))])
                    idx_val.extend(idx_cl[int((1 - val_ratio) * (1 - test_ratio) * len(idx_cl)):int((1 - test_ratio) * len(idx_cl))])
                    idx_test.extend(idx_cl[int((1 - test_ratio) * len(idx_cl)):])

        f_train = mimii.get_filepaths(idx_train)
        f_val = mimii.get_filepaths(idx_val)
        f_test = mimii.get_filepaths(idx_test)
        return f_train, f_val, f_test

    def get_all_ids(self):
        return np.arange(self.__len__()) #np.array([self.get_target(idx) for idx in range(self.__len__())])

    def get_filepaths(self, idxs):
        idxs = np.array(idxs)
        idxs = idxs % self.total_samples
        return self.data.loc[idxs, 'filepath'].tolist()

    def get_snrs(self, idxs):
        idxs = idxs % self.total_samples
        return self.data.loc[idxs, 'snr'].tolist()

    def get_target(self, idx):
        idx = idx % self.total_samples
        target = [self.data[t][idx] for t in self.target]
        return np.sum(np.array(np.array(target, dtype=np.str), dtype=np.object), axis=0)

    def get_waveform(self, idx):
        idx = idx % self.total_samples
        y, sr = librosa.load(self.data['filepath'][idx], sr=None, mono=False)
        assert sr == self.samplerate
        return y[self.channels]

    def __getitem__(self, idx):
        # if idx > 0: raise IndexError
        # y, sr = librosa.load(self.data['filepath'][idx], sr=None, mono=False)
        segment = idx // self.total_samples
        idx = idx % self.total_samples
        sr, y = wavread(self.data['filepath'][idx])
        y = np.transpose(np.array(y, dtype=np.float32)) / 32768.0
        # assert sr == self.samplerate
        target = [self.data[t][idx] for t in self.target]
        target = np.array(target, dtype=np.str)
        target = np.sum(np.array(target, dtype=np.object), axis=0)
        return torch.from_numpy(y[self.channels, int(self.wav_length * segment):int(self.wav_length * (segment+1))]), torch.from_numpy(np.array(self.one_hot_encode([target]), dtype=np.float32)[0])


