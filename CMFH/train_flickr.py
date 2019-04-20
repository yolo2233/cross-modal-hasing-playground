import numpy as np
import flickr
import scipy.io as sio
from cmfh import CMFH
import time

IMAGE_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_vgg19_20015.npy'
TEXT_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_bow_20015.npy'

hash_len = 64
with open('results/results.txt', 'a', encoding='utf-8') as f:
    f.write('*********************************************************\n')
    f.write('Script run at ' + time.strftime('%Y-%m-%d %H:%M:%S\n'))
    f.write('dataset: flickr25k')
    f.write('train on 5000 instances, hash bit length ' + str(hash_len) + '\n')


query_list, retrieval_list, training_list = flickr.sample(query_num=1000, training_num=5000)
training_text = flickr.sample_feature(training_list, TEXT_FEATURE, modal='bow')
training_image = flickr.sample_feature(training_list, IMAGE_FEATURE)

query_image = flickr.sample_feature(query_list, IMAGE_FEATURE)
retrieval_image = flickr.sample_feature(retrieval_list, IMAGE_FEATURE)
query_text = flickr.sample_feature(query_list, TEXT_FEATURE, modal='bow')
retrieval_text = flickr.sample_feature(retrieval_list, TEXT_FEATURE, modal='bow')
simi_matrix = flickr.generate_similarity_matrix(query_list, retrieval_list)

cmfh = CMFH(hash_len)
cmfh.train(training_image, training_text)
cmfh.eval(query_image, query_text, retrieval_image, retrieval_text, simi_matrix)
