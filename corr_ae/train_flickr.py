import os
import time
import logging
import numpy as np

from utils import check_folder
import model
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

LATENT_LEN = 128
ITERATION = 3001
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
METRIC = 'hash'
RADIUS = None

if METRIC == 'hash':
    RADIUS = 4

DATA_ROOT_PATH = 'data/'
RESULT_ROOT_PATH = 'results/'
SAVE_MODEL_ROOT_PATH = 'saved_model/'
LOG_ROOT_PATH = 'log/'
SNAPSHOT = 'snapshots/'

IMAGE_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_vgg19_20015.npy'
TEXT_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_bow_20015.npy'
LABEL_PATH = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_label_20015.npy'

check_folder(DATA_ROOT_PATH)
check_folder(RESULT_ROOT_PATH)
check_folder(SAVE_MODEL_ROOT_PATH)
check_folder(LOG_ROOT_PATH)
check_folder(SNAPSHOT)

logging.basicConfig(filename=LOG_ROOT_PATH + 'log.log')

with open(RESULT_ROOT_PATH + 'result.txt', 'a', encoding='utf-8') as f:
    f.write('==================================================================\n')
    f.write('Script run at ' + time.strftime('%Y-%m-%d %H:%M:%S\n'))
    f.write('Latent code length: ' + str(LATENT_LEN) + '\n')
    f.write('dataset: flickr\n')
    if METRIC == 'hash':
        f.write('Radius: ' + str(RADIUS) + '\n')

# prepare the data
query_list, retrieval_list, training_list = utils.sample(query_num=1000, training_num=5000)
training_image = utils.sample_feature(training_list, IMAGE_FEATURE)
training_text = utils.sample_feature(training_list, TEXT_FEATURE, modal='bow')

similarity_matrix = utils.generate_similarity_matrix(query_list, retrieval_list, LABEL_PATH)

query_image = utils.sample_feature(query_list, IMAGE_FEATURE)
query_text = utils.sample_feature(query_list, TEXT_FEATURE, 'bow')
retrieval_image = utils.sample_feature(retrieval_list, IMAGE_FEATURE)
retrieval_text = utils.sample_feature(retrieval_list, TEXT_FEATURE, 'bow')

data = np.concatenate((training_image, training_text), axis=1)

train_feed_dict = {
    'lr': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'snapshot': SNAPSHOT,
    'data': data,
    'iteration': ITERATION,
    'keep_prob': 0.5
}

test_feed_dict = {
    'image_query': query_image,
    'image_re': retrieval_image,
    'text_query': query_text,
    'text_re': retrieval_text,
    'similarity_matrix': similarity_matrix,
    'metric': 'hash',
    'radius': RADIUS
}

# try:
# training model
Corr_AE = model.Corr_AE(4096, 1386)
Corr_AE.build_graph(LATENT_LEN)
Corr_AE.train(train_feed_dict, test_feed_dict)

# eval


# MAP_i2t, precision_i2t, *pr_i2t = Corr_AE.eval(query_image, retrieval_text, similarity_matrix, 'img', 'txt', METRIC, RADIUS)
# MAP_t2i, precision_t2i, *pr_t2i = Corr_AE.eval(query_text, retrieval_image, similarity_matrix, 'txt', 'img', METRIC, RADIUS)
#
# with open(RESULT_ROOT_PATH + 'result.txt', 'a', encoding='utf-8') as f:
#
#     f.write('top_k_precision_i2t: 10, 20, 50, 100, 500: ' + str(precision_i2t) + '\n')
#     f.write('MAP_i2t: ' + str(MAP_i2t) + '\n\n')
#     if pr_i2t is not None:
#         MAP_comp, top_k_precision_comp, precision, recall = pr_i2t
#         f.write('cosine MAP_i2t:' + str(MAP_comp) + '\n')
#         f.write('cosine top_k_precision_i2t:' + str(top_k_precision_comp) + '\n')
#         f.write('precision_i2t: ' + str(precision) + '\n')
#         f.write('recall_i2t: ' + str(recall) + '\n\n')
#
#     f.write('top_k_precision_t2i: 10, 20, 50, 100, 500: ' + str(precision_t2i) + '\n')
#     f.write('MAP_t2i: ' + str(MAP_t2i) + '\n\n')
#     if pr_t2i is not None:
#         MAP_comp, top_k_precision_comp, precision, recall = pr_i2t
#         f.write('cosine MAP_t2i:' + str(MAP_comp) + '\n')
#         f.write('cosine top_k_precision_t2i:' + str(top_k_precision_comp) + '\n')
#         f.write('precision_t2i: ' + str(precision) + '\n')
#         f.write('recall_t2i: ' + str(recall) + '\n\n')
#
