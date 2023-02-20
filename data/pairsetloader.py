import math
from tensorflow.keras.utils import Sequence
from utils import *
import os
import cv2
import tensorflow as tf
import numpy

class PairSetDataLoader(Sequence):
    def __init__(self, data_list, batch_size, shuffle=False):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.data_list) / self.batch_size)

    # batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
        # sampler 의 역할 (index를 batch_size 만큼 sampling 해줌)
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = np.array([self.get_img(self.data_list[i]) for i in indices])
        batch_ref = np.expand_dims(batch_data[:, :, :, 0], axis=-1)
        batch_test = np.expand_dims(batch_data[:, :, :, 1], axis=-1)
        batch_label = np.expand_dims(batch_data[:, :, :, 2], axis=-1)

        return (tf.convert_to_tensor(batch_ref), tf.convert_to_tensor(batch_test)), tf.convert_to_tensor(batch_label)

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.seed()
            np.random.shuffle(self.indices)

    def get_img(self, single_data_list):
        img = cv2.imread(single_data_list, cv2.IMREAD_GRAYSCALE)

        ref_img = np.float32(img[:, :img.shape[0] * 1]) / 255
        test_img = np.float32(img[:, img.shape[0]:img.shape[0] * 2]) / 255
        ref_label = img[:, img.shape[0] * 2:img.shape[0] * 3] // 255
        test_label = img[:, img.shape[0] * 3:img.shape[0] * 4] // 255
        comb_label = np.bitwise_or(ref_label, test_label)

        return np.concatenate((ref_img[:, :, np.newaxis], test_img[:, :, np.newaxis], comb_label[:, :, np.newaxis]), -1)

    def __get_file_name__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_file_name = np.array([os.path.basename(self.data_list[i])[:-4] for i in indices])
        return batch_file_name






