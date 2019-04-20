import os
import time
import logging
import numpy as np

from utils import check_folder
import model
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

LATENT_LEN = 256
ITERATION = 2501
BATCH_SIZE = 128
LEARNING_RATE = 0.001
METRIC = 'hash'



# DATA_ROOT_PATH = 'data/'
RESULT_PATH = 'results/'
SAVE_MODEL_ROOT_PATH = 'saved_model/'
# LOG_ROOT_PATH = 'log/'
SNAPSHOT = 'snapshots/snap_shots_flickr/'

IMAGE_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_vgg19_20015.npy'
TEXT_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_bow_20015.npy'
LABEL_PATH = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_label_20015.npy'

# check_folder(DATA_ROOT_PATH)
check_folder(SAVE_MODEL_ROOT_PATH)
# check_folder(LOG_ROOT_PATH)
check_folder(SNAPSHOT)
check_folder(RESULT_PATH)

# logging.basicConfig(filename=LOG_ROOT_PATH + 'log.log')

with open(RESULT_PATH + 'result_flickr.txt', 'a', encoding='utf-8') as f:
    f.write('==================================================================\n')
    f.write('Script run at ' + time.strftime('%Y-%m-%d %H:%M:%S\n'))
    f.write('Latent code length: ' + str(LATENT_LEN) + '\n')
    f.write('dataset: flickr\n')

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
}


dbrc= model.DBRC(4096, 1386, LATENT_LEN)
dbrc.build_graph()
dbrc.train(train_feed_dict, test_feed_dict, RESULT_PATH + 'result_flickr.txt')

