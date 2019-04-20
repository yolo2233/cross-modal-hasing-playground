import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import scipy.io as sio
import random

def count_time(func):
    def wrapper(*args, **kwargs):
        with open('results/results.txt', 'a', encoding='utf-8') as f:
            f.write('excute ' + func.__name__ + '...\n')
        tic = time.clock()
        results = func(*args, **kwargs)
        toc = time.clock()
        with open('results/results.txt', 'a', encoding='utf-8') as f:
            f.write(func.__name__ + ' cost time: ' + str(round(toc - tic, 4)) + '\n\n')

        return results

    return wrapper


def sample(query_num, training_num, dataset_size):
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


def optimized_mAP(similarity_matrix, modal='i2t', top=None):
    ham_dis = sio.loadmat('temp_data/ham_dis.mat')
    distance_i2t = ham_dis['ham_dis_i2t']
    distance_t2i = ham_dis['ham_dis_t2i']
    query_size = distance_i2t.shape[0]

    if modal == 'i2t':
        distance = distance_i2t
    else:
        distance = distance_t2i

    sorted_index = np.argsort(distance)

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


def precision_recall(similarity_matrix, modal='i2t'):
    ham_dis = sio.loadmat('temp_data/ham_dis.mat')
    distance_i2t = ham_dis['ham_dis_i2t']
    distance_t2i = ham_dis['ham_dis_t2i']

    if modal == 'i2t':
        distance = distance_i2t
    else:
        distance = distance_t2i

    pre_list = []
    recall_list = []

    for radius in [i for i in range(33) if i % 2 == 0 and i > 0]:
        temp_distance = distance.copy()
        temp_distance[temp_distance <= radius] = 1
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


def precision_top_k( similarity_matrix, top_k: list, modal='i2t') -> list:
    # calculate top k precision
    ham_dis = sio.loadmat('temp_data/ham_dis.mat')
    distance_i2t = ham_dis['ham_dis_i2t']
    distance_t2i = ham_dis['ham_dis_t2i']

    if modal == 'i2t':
        distance = distance_i2t
    else:
        distance = distance_t2i

    sorted_index = np.argsort(distance)

    sorted_simi_matrix = np.array(list(map(lambda x, y: x[y], similarity_matrix, sorted_index)))
    precision = []
    for i in top_k:
        average_precison_top_i = np.mean(np.sum(sorted_simi_matrix[:, :i], axis=1) / i)
        precision.append(average_precison_top_i)

    precision = [round(i, 4) for i in precision]
    return precision


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