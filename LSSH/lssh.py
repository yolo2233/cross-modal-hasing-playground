from sklearn.preprocessing import Normalizer
import numpy as np
import matlab
import matlab.engine
import utils
import scipy.io as sio
import os
import time


class LSSH(object):

    def __init__(self, hash_len, train_img, train_txt, query_image, query_text, retrieval_image, retrieval_text):
        if not os.path.exists('temp_data'):
            os.mkdir('temp_data')

        # normalize data
        norm2 = Normalizer(norm='l2')
        train_img = norm2.fit_transform(train_img)
        train_txt = norm2.fit_transform(train_txt)
        query_image = norm2.fit_transform(query_image)
        query_text = norm2.fit_transform(query_text)
        retrieval_image = norm2.fit_transform(retrieval_image)
        retrieval_text = norm2.fit_transform(retrieval_text)

        sio.savemat('temp_data/flickr_data.mat', {'train_image': np.transpose(train_img),
                                                  'train_text': np.transpose(train_txt),
                                                  'query_image': query_image,
                                                  'query_text': query_text,
                                                  'retrieval_image': retrieval_image,
                                                  'retrieval_text': retrieval_text})

        self.flickr_data = sio.loadmat('temp_data/flickr_data.mat')
        self.hash_len = hash_len

    @utils.count_time
    def train(self):
        print('training...')
        engine = matlab.engine.start_matlab()
        engine.train_lssh(matlab.double([self.hash_len]), nargout=0)

    @utils.count_time
    def eval(self, simi_matrix):
        print('evaluating...')
        hash_code = sio.loadmat('temp_data/hash_code.mat')
        q_img_hash = np.transpose(hash_code['query_image_hash'])
        q_txt_hash = np.transpose(hash_code['query_text_hash'])
        r_img_hash = np.transpose(hash_code['retrieval_image_hash'])
        r_txt_hash = np.transpose(hash_code['retrieval_text_hash'])

        # i2t_map = utils.optimized_mAP(q_img_hash, r_txt_hash, simi_matrix, 'hash', top=1000)
        # i2t_pre_top_k = utils.precision_top_k(q_img_hash, r_txt_hash, simi_matrix,
        #                                       [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        #                                        800, 850, 900, 950, 1000], 'hash')
        i2t_pre, i2t_recall = utils.precision_recall(q_img_hash, r_txt_hash, simi_matrix)

        # t2i_map = utils.optimized_mAP(q_txt_hash, r_img_hash, simi_matrix, 'hash', top=1000)
        # t2i_pre_top_k = utils.precision_top_k(q_txt_hash, r_img_hash, simi_matrix,
        #                                       [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        #                                        800, 850, 900, 950, 1000], 'hash')
        t2i_pre, t2i_recall = utils.precision_recall(q_txt_hash, r_img_hash, simi_matrix)

        with open('results/results.txt', 'a', encoding='utf-8') as f:
            # f.write('i2t map: ' + str(round(i2t_map, 4)) + '\n')
            # f.write(
            #     'i2t top [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] precision: ' + str(
            #         i2t_pre_top_k) + '\n')
            f.write('i2t precision: ' + str(i2t_pre) + '\n')
            f.write('t2i recall: ' + str(i2t_recall) + '\n')
            # f.write('t2i map: ' + str(round(t2i_map, 4)) + '\n')
            # f.write(
            #     't2i top [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] precision: ' + str(
            #         t2i_pre_top_k) + '\n')
            f.write('t2i precision: ' + str(t2i_pre) + '\n')
            f.write('t2i recall: ' + str(t2i_recall) + '\n\n')
