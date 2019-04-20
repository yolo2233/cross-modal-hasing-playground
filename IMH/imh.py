from sklearn.preprocessing import Normalizer
import numpy as np
import matlab
import matlab.engine
import utils
import scipy.io as sio
import os


class IMH:
    def __init__(self, hash_len, train_img, train_txt, query_img, query_txt, retrieval_img, retrieval_txt):
        if not os.path.exists('temp_data'):
            os.mkdir('temp_data')

        self.training_num = train_img.shape[0]
        self.hash_len = hash_len
        self.img_mean = np.mean(train_img, axis=0, keepdims=True)
        self.txt_mean = np.mean(train_txt, axis=0, keepdims=True)

        self.train_img = train_img - self.img_mean
        self.train_txt = train_txt - self.txt_mean
        self.query_img = query_img - self.img_mean
        self.retrieval_img = retrieval_img - self.img_mean
        self.query_txt = query_txt - self.txt_mean
        self.retrieval_txt = retrieval_txt - self.txt_mean

        sio.savemat('temp_data/flickr_data.mat', {'train_img': self.train_img,
                                                  'train_txt': self.train_txt,
                                                  'query_img': self.query_img,
                                                  'query_txt': self.query_txt,
                                                  'retrieval_img': self.retrieval_img,
                                                  'retrieval_txt': self.retrieval_txt})

    @utils.count_time
    def train(self):
        print('training...')
        engine = matlab.engine.start_matlab()
        engine.trainIMH(matlab.double([self.training_num]), matlab.double([self.hash_len]), nargout=0)

    @utils.count_time
    def eval(self, simi_matrix):
        print('evaluating...')

        # i2t_map = utils.optimized_mAP(simi_matrix, 'i2t')
        # i2t_pre_top_k = utils.precision_top_k(simi_matrix, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], 'i2t')
        i2t_pre, i2t_recall = utils.precision_recall(simi_matrix, modal='i2t')

        # t2i_map = utils.optimized_mAP(simi_matrix, 't2i')
        # t2i_pre_top_k = utils.precision_top_k(simi_matrix, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], 't2i')
        t2i_pre, t2i_recall = utils.precision_recall(simi_matrix, modal='t2i')

        with open('results/results.txt', 'a', encoding='utf-8') as f:
            # f.write('i2t map: ' + str(round(i2t_map, 4)) + '\n')
            # f.write('i2t top [[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] precision: ' + str(i2t_pre_top_k) + '\n')
            # f.write('i2t precision: %.5f, i2t recall: %.5f: \n' % (i2t_pre, i2t_recall))
            # f.write('t2i map: ' + str(round(t2i_map, 4)) + '\n')
            # f.write('t2i top [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] precision: ' + str(t2i_pre_top_k) + '\n')
            # f.write('t2i precision: %.5f, t2i recall: %.5f: \n\n' % (t2i_pre, t2i_recall))
            f.write('i2t precision: ' + str(i2t_pre) + '\n')
            f.write('t2i recall: ' + str(i2t_recall) + '\n')

            f.write('t2i precision: ' + str(t2i_pre) + '\n')
            f.write('t2i recall: ' + str(t2i_recall) + '\n\n')
