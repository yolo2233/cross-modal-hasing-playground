import tensorflow as tf
import numpy as np
import time
import os
import h5py
import logging
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import random


def count_time(result_file):
    def wrapper(func):
        def return_wrapper(*args, **kwargs):
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write('excute ' + func.__name__ + '...\n')
            tic = time.clock()
            results = func(*args, **kwargs)
            toc = time.clock()
            with open('results/result_flickr.txt', 'a', encoding='utf-8') as f:
                f.write(func.__name__ + ' cost time: ' + str(toc - tic) + '\n\n')

            return results
        return return_wrapper
    return wrapper



def sample(query_num, training_num, dataset_size=20015):
    '''
    从数据集里随机选取query_num个作为查询集
    剩下的点作为被查询集，从被查询集中选出training_num个点作为训练集

    '''
    sample_list = [i for i in range(dataset_size)]
    query_list = random.sample(sample_list, query_num)
    retrieval_list = list(set(sample_list) - set(query_list))
    training_list = random.sample(retrieval_list, training_num)

    return query_list, retrieval_list, training_list


def generate_similarity_matrix(query_list, retrieval_list, label_path):
    labels = np.load(label_path)
    dim = labels.shape[1]
    query_label = np.zeros((len(query_list), dim))
    retrieval_label = np.zeros((len(retrieval_list), dim))


    for index, i in enumerate(query_list):
        query_label[index, ...] = labels[i, ...]

    for index, i in enumerate(retrieval_list):
        retrieval_label[index, ...] = labels[i, ...]

    query_label = np.expand_dims(query_label.astype(dtype=np.int), 1)
    retrieval_label = np.expand_dims(retrieval_label.astype(dtype=np.int), 0)
    similarity_matrix = np.bitwise_and(query_label, retrieval_label)
    similarity_matrix = np.sum(similarity_matrix, axis=2)
    similarity_matrix[similarity_matrix >= 1] = 1

    return similarity_matrix


def sample_feature(sample_list, path, modal='image'):
    '''
    用来从vgg feature和bow 中进行采样
    '''
    data = np.load(path)  # load .npy file

    dim = data.shape[1]
    if modal == 'image':
        feature = np.zeros((len(sample_list), dim))
    elif modal == 'word2vec':
        feature = np.zeros((len(sample_list), dim))
    else:
        feature = np.zeros((len(sample_list), dim))

    for index, i in enumerate(sample_list):
        feature[index, ...] = data[i, ...]

    return feature


def conv2d(input, output_dim, name, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, padding='SAME', activation='relu',
           batch_norm=True, training=True):
    input_dim = input.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('kernel', [k_h, k_w, input_dim, output_dim],
                                 initializer=tf.initializers.truncated_normal(stddev=stddev))
        conv = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.initializers.constant(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=training)  # training is False when test
        if activation == 'relu':
            conv = tf.nn.relu(conv)
        elif activation == 'tanh':
            conv = tf.nn.tanh(conv)
        elif activation == 'sigmoid':
            conv = tf.nn.sigmoid(conv)

        return conv


def de_conv(input, output_shape, name, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, padding='SAME', activation='relu',
            batch_norm=True, training=True):
    # 对于纯卷积层的反卷积的方法是 先conv2d_nn_transpose
    # 在加上bias 在activation
    # filter : [height, width, output_channels, in_channels]
    input_dim = input.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('kernel', [k_h, k_w, output_shape[-1], input_dim],
                                 initializer=tf.initializers.random_normal(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, s_h, s_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.initializers.constant(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv = tf.reshape(deconv, deconv.get_shape())

        if batch_norm:
            deconv = tf.layers.batch_normalization(deconv, training=training)
        if activation == 'relu':
            deconv = tf.nn.relu(deconv)
        elif activation == 'tanh':
            deconv = tf.nn.tanh(deconv)
        elif activation == 'sigmoid':
            deconv = tf.nn.sigmoid(deconv)

        return deconv


def fully_connected(input, output_dim, name, stddev=0.02, bias_init=0.0, activation='relu', batch_norm=True,
                    training=True, keep_prob=1.0, aug=False):
    input_dim = input.get_shape()[1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [input_dim, output_dim], tf.float32,
                                  tf.initializers.random_normal(stddev=stddev))
        bias = tf.get_variable('bias', [output_dim],
                               initializer=tf.initializers.constant(bias_init))

        fc = tf.matmul(input, weights) + bias

        if aug:
            fc = fc * 100

        if batch_norm:
            fc = tf.layers.batch_normalization(fc, training=training)  # training is False when test
        if activation == 'relu':
            fc = tf.nn.relu(fc)
        elif activation == 'tanh':
            fc = tf.nn.tanh(fc)
        elif activation == 'sigmoid':
            fc = tf.nn.sigmoid(fc)
        elif activation == 'softmax':
            fc = tf.nn.softmax(fc)

        fc = tf.nn.dropout(fc, keep_prob=keep_prob)

        return fc


def hamming_distance(X, Y):
    '''
    返回两个矩阵以行为pair的汉明距离
    :param X: (n, hash_len)
    :param Y: (m, hash_len)
    :return: (n, m)
    '''

    res = np.bitwise_xor(np.expand_dims(X, 1), np.expand_dims(Y, 0))
    res = np.sum(res, axis=2)
    return res


def kl_divergence(p, q):
    return tf.reduce_sum(p * tf.log(p / q), axis=1)


def optimized_mAP(q, r, similarity_matrix, dis_metric='hash', top=None):
    query = q.copy()
    retrieval = r.copy()

    query_size = query.shape[0]

    if dis_metric == 'hash':

        query[query >= 0] = 1
        query[query != 1] = 0
        retrieval[retrieval >= 0] = 1
        retrieval[retrieval != 1] = 0

        query = query.astype(dtype=np.int8)
        retrieval = retrieval.astype(dtype=np.int8)
        distance = hamming_distance(query, retrieval)
    elif dis_metric == 'eu':
        distance = euclidean_distances(query, retrieval)
    else:
        distance = cosine_similarity(query, retrieval)

    sorted_index = np.argsort(distance)
    if dis_metric == 'cosine':
        sorted_index = np.flip(sorted_index, axis=1)

    sorted_similarity_matrix = np.array(list(map(lambda x, y: x[y], similarity_matrix, sorted_index)))
    sorted_similarity_matrix = np.asarray(sorted_similarity_matrix)[:, :top]
    neighbors = np.sum(sorted_similarity_matrix, axis=1)
    one_index = np.argwhere(sorted_similarity_matrix == 1)
    precision = 0
    cnt = 0

    for i in range(query_size):
        precision_at_i = 0
        if neighbors[i] == 0:
            continue
        for j in range(neighbors[i]):
            precision_at_i += np.sum(sorted_similarity_matrix[i, :one_index[cnt, 1] + 1]) / (one_index[cnt, 1] + 1)
            cnt += 1
        precision += precision_at_i / neighbors[i]
    mAP = precision / query_size

    return mAP


def precision_recall(q, r, similarity_matrix):
    query = q.copy()
    retrieval = r.copy()

    query[query >= 0] = 1
    query[query != 1] = 0
    retrieval[retrieval >= 0] = 1
    retrieval[retrieval != 1] = 0

    query = query.astype(dtype=np.int8)
    retrieval = retrieval.astype(dtype=np.int8)

    distance = hamming_distance(query, retrieval)

    pre_list = []
    recall_list = []

    for radius in [i for i in range(33) if i % 2 == 0 and i > 0]:
        temp_distance = distance.copy()
        temp_distance[distance <= radius] = 1
        temp_distance[temp_distance > radius] = 0

        tp = np.sum(similarity_matrix * temp_distance)
        precision = 0
        recall = 0
        if tp != 0:
            precision = tp / np.sum(temp_distance)
            recall = tp / np.sum(similarity_matrix)
        pre_list.append(precision)
        recall_list.append(recall)

    pre_list = [round(i, 4) for i in pre_list]
    recall_list = [round(i, 4) for i in recall_list]

    return pre_list, recall_list


def precision_top_k(q, r, similarity_matrix, top_k: list, dis_metric) -> list:
    # calculate top k precision
    query = q.copy()
    retrieval = r.copy()

    if dis_metric == 'hash':

        query[query >= 0] = 1
        query[query != 1] = 0
        retrieval[retrieval >= 0] = 1
        retrieval[retrieval != 1] = 0

        query = query.astype(dtype=np.int8)
        retrieval = retrieval.astype(dtype=np.int8)
        distance = hamming_distance(query, retrieval)
    elif dis_metric == 'eu':
        distance = euclidean_distances(query, retrieval)
    else:
        distance = cosine_similarity(query, retrieval)

    sorted_index = np.argsort(distance)
    if dis_metric == 'cosine':
        sorted_index = np.flip(sorted_index, axis=1)

    sorted_simi_matrix = np.array(list(map(lambda x, y: x[y], similarity_matrix, sorted_index)))
    precision = []
    for i in top_k:
        average_precison_top_i = np.mean(np.sum(sorted_simi_matrix[:, :i], axis=1) / i)
        precision.append(average_precison_top_i)

    precision = [round(i, 4) for i in precision]
    return precision


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def data_reader(path):
    data = h5py.File(path, 'r+')
    key = list(data.keys())[0]
    group = data[key]
    return group.value


if __name__ == '__main__':
    import time

    q_set = np.array([[1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1]])
    r_set = np.array([[1, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 1]])
    simi = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])

    # anno_dict = {1: {0, 1}, 3: {3, 4}, 4: {0, 5}, 7: {4, 3}, 8: {4, 8}, 9: {1, 3}}
    # m = mAP(q_set, r_set, q_set_list, r_set_list, anno_dict)
    m1 = optimized_mAP(q_set, r_set, simi, 1)
    print(m1)
    '''
    人工计算：
        Q = 3
        i==1
        m_1 == 2
            p = (1 + 1) / 2  = 1
        i == 2
        m_2 == 2
            p =  (0 + 0.5) / 2 = 0.25
        i == 3
        m_3 == 2
            p = (0 + 0.5) / 2 = 0.25

        mAP = (0.25 + 0.25 + 1) / 3 = 0.5/3 = 0.5     

    '''
    # q_set = np.array([[1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 1]])
    # r_set = np.array([[1, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 0]])
    # simi = np.array([
    #     [0, 1, 0, 0],
    #     [0, 1, 1, 1],
    #     [0, 1, 1, 1]
    # ])
    # q_set_list = [1, 3, 7]
    # r_set_list = [4, 9, 8, 2]
    # anno_dict = {1: {0, 1}, 2: {3, 8}, 3: {3, 4}, 4: {2, 5}, 7: {4, 3}, 8: {4, 8}, 9: {1, 3}}

    '''
        Q = 3
        i == 1
        m_1 == 1
        p = 0/1 = 0

        i ==2
        m_2 == 3
        p = (0 + 0.5 + 2/3)/ 3 = 0.38888888

        i == 3
        m_3 == 3
        p = (0 + 0.5 + 2/3)/3 = 0.38888888

        mAP = (0.3888*2 + 0)/3 = 0.259259

    '''
    # m = mAP(q_set, r_set, q_set_list, r_set_list, anno_dict)
    # m1 = optimized_mAP(q_set, r_set, simi)
