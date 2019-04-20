import numpy as np
import utils
import tensorflow as tf

class CMFH(object):
    def __init__(self, hash_len):
        self.U_1 = np.random.rand(4096, hash_len)
        self.U_2 = np.random.rand(1386, hash_len)
        self.P_1 = np.random.rand(hash_len, 4096)
        self.P_2 = np.random.rand(hash_len, 1386)
        self.V = None

        self.lamb = 0.5
        self.mu = 100
        self.gamma = 0.001
        self.hash_len = hash_len

        self.img_mean = None
        self.txt_mean = None

    @utils.count_time
    def train(self, train_img, train_txt):
        # center data
        self.img_mean = np.mean(train_img, axis=0, keepdims=True)
        self.txt_mean = np.mean(train_txt, axis=0, keepdims=True)

        train_img = np.transpose(train_img - self.img_mean)
        train_txt = np.transpose(train_txt - self.txt_mean)
        min_loss = 9999999999
        cnt = 0

        while True:
            # update V
            temp_V_left = self.lamb * np.matmul(np.transpose(self.U_1), self.U_1) + \
                          self.lamb * np.matmul(np.transpose(self.U_2), self.U_2) + \
                          (2 * self.mu + self.gamma) * np.eye(self.hash_len)

            temp_V_left = np.linalg.inv(temp_V_left)  # hash_len * hash_len

            temp_V_right = np.matmul(self.lamb * np.transpose(self.U_1) + self.mu * self.P_1, train_img) + \
                           np.matmul(self.lamb * np.transpose(self.U_2) + self.mu * self.P_2,
                                     train_txt)  # hash_len * num
            self.V = np.matmul(temp_V_left, temp_V_right)

            # update P
            temp_P_1_left = np.matmul(self.V, np.transpose(train_img))  # hash_len x 4096
            temp_P_1_right = np.linalg.inv(
                np.matmul(train_img, np.transpose(train_img)) + (self.gamma / self.mu) * np.eye(4096, 4096)
            )  # 4096 x 4096

            self.P_1 = np.matmul(temp_P_1_left, temp_P_1_right)

            temp_P_2_left = np.matmul(self.V, np.transpose(train_txt))  # hash_len x 4096
            temp_P_2_right = np.linalg.inv(
                np.matmul(train_txt, np.transpose(train_txt)) + (self.gamma / self.mu) * np.eye(1386, 1386)
            )  # 4096 x 4096

            self.P_2 = np.matmul(temp_P_2_left, temp_P_2_right)

            # update U
            temp_U_1_left = np.matmul(train_img, np.transpose(self.V))  # 4096 * hash_len
            temp_U_1_right = np.linalg.inv(
                np.matmul(self.V, np.transpose(self.V)) + (self.gamma / self.lamb) * np.eye(self.hash_len,
                                                                                            self.hash_len)
            )  # hash_len * hash_len
            self.U_1 = np.matmul(temp_U_1_left, temp_U_1_right)

            temp_U_2_left = np.matmul(train_txt, np.transpose(self.V))  # 4096 * hash_len
            temp_U_2_right = np.linalg.inv(
                np.matmul(self.V, np.transpose(self.V)) + (self.gamma / self.lamb) * np.eye(self.hash_len,
                                                                                            self.hash_len)
            )  # hash_len * hash_len
            self.U_2 = np.matmul(temp_U_2_left, temp_U_2_right)

            loss = self.lamb * np.linalg.norm(train_img - np.matmul(self.U_1, self.V), 'fro') ** 2 + \
                   self.lamb * np.linalg.norm(train_txt - np.matmul(self.U_2, self.V), 'fro') ** 2+ \
                   self.mu * (
                           np.linalg.norm(self.V - np.matmul(self.P_1, train_img), 'fro') ** 2 +
                           np.linalg.norm(self.V - np.matmul(self.P_2, train_txt), 'fro') ** 2) + \
                   self.gamma * (np.linalg.norm(self.V, 'fro') ** 2 +
                                 np.linalg.norm(self.P_1, 'fro') ** 2 +
                                 np.linalg.norm(self.P_2, 'fro') ** 2 +
                                 np.linalg.norm(self.U_1, 'fro') ** 2 +
                                 np.linalg.norm(self.U_2, 'fro') ** 2)

            if cnt % 10 == 0:
                print('at iteration %d, loss is: %.6f, min loss is: %.6f' % (cnt, loss, min_loss))

            cnt += 1

            if min_loss - loss < 0.01:
                print('converged...')
                print('min loss: %.5f \t loss: %.5f' % (min_loss, loss))
                break

            min_loss = loss

    @utils.count_time
    def eval(self, query_img, query_txt, retrieval_img, retrieval_txt, simi_matrix):
        query_img = query_img - self.img_mean
        retrieval_img = retrieval_img - self.img_mean
        query_txt = query_txt - self.txt_mean
        retrieval_txt = retrieval_txt - self.txt_mean

        P_1 = np.transpose(self.P_1)
        q_img_hash = np.sign(np.matmul(query_img, P_1))
        r_img_hash = np.sign(np.matmul(retrieval_img, P_1))

        P_2 = np.transpose(self.P_2)
        q_txt_hash = np.sign(np.matmul(query_txt, P_2))
        r_txt_hash = np.sign(np.matmul(retrieval_txt, P_2))

        i2t_map = utils.optimized_mAP(q_img_hash, r_txt_hash, simi_matrix, 'hash')
        i2t_pre_top_k = utils.precision_top_k(q_img_hash, r_txt_hash, simi_matrix, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], 'hash')
        i2t_pre, i2t_recall = utils.precision_recall(4, q_img_hash, r_txt_hash, simi_matrix)

        t2i_map = utils.optimized_mAP(q_txt_hash, r_img_hash, simi_matrix, 'hash')
        t2i_pre_top_k = utils.precision_top_k(q_txt_hash, r_img_hash, simi_matrix, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], 'hash')
        t2i_pre, t2i_recall = utils.precision_recall(4, q_txt_hash, r_img_hash, simi_matrix)

        with open('results/results.txt', 'a', encoding='utf-8') as f:
            f.write('i2t map: ' + str(round(i2t_map, 4)) + '\n')
            f.write('i2t top [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] precision: ' + str(i2t_pre_top_k) + '\n')
            f.write('i2t precision: %.5f, i2t recall: %.5f: \n' % (i2t_pre, i2t_recall))
            f.write('t2i map: ' + str(round(t2i_map, 4)) + '\n')
            f.write('t2i top [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] precision: ' + str(t2i_pre_top_k) + '\n')
            f.write('t2i precision: %.5f, t2i recall: %.5f \n\n' % (t2i_pre, t2i_recall))
