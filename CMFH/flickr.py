import numpy as np
import random

LABEL_PATH = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_label_20015.npy'


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


def generate_similarity_matrix(query_list, retrieval_list, label_path=LABEL_PATH):
    query_label = np.zeros((len(query_list), 24))
    retrieval_label = np.zeros((len(retrieval_list), 24))

    labels = np.load(label_path)

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
    if modal == 'image':
        feature = np.zeros((len(sample_list), 4096))
    elif modal == 'word2vec':
        feature = np.zeros((len(sample_list), 300))
    else:
        feature = np.zeros((len(sample_list), 1386))

    for index, i in enumerate(sample_list):
        feature[index, ...] = data[i, ...]

    return feature
