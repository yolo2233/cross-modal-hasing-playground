import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


def count_time(func):
    def wrapper(*args, **kwargs):
        with open('results/results.txt', 'a', encoding='utf-8') as f:
            f.write('excute ' + func.__name__ + '...\n')
        tic = time.clock()
        results = func(*args, **kwargs)
        toc = time.clock()
        with open('results/results.txt', 'a', encoding='utf-8') as f:
            f.write(func.__name__ + ' cost time: ' + str(toc - tic) + '\n\n')

        return results

    return wrapper


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


def precision_recall(radius, q, r, similarity_matrix):
    query = q.copy()
    retrieval = r.copy()

    query[query >= 0] = 1
    query[query != 1] = 0
    retrieval[retrieval >= 0] = 1
    retrieval[retrieval != 1] = 0

    query = query.astype(dtype=np.int8)
    retrieval = retrieval.astype(dtype=np.int8)

    distance = hamming_distance(query, retrieval)

    distance[distance <= radius] = 1
    distance[distance > radius] = 0

    tp = np.sum(similarity_matrix * distance)
    precision = 0
    recall = 0
    if tp != 0:
        precision = tp / np.sum(distance)
        recall = tp / np.sum(similarity_matrix)

    return precision, recall


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