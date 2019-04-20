from imh import IMH
import time
import utils

IMAGE_FEATURE = '../../dataset/nuswide/nuswide_image.npy'
TEXT_FEATURE = '../../dataset/nuswide/nuswide_txt.npy'
LABEL_PATH = '../../dataset/nuswide/nuswide_label.npy'



hash_len = 32

with open('results/results.txt', 'a', encoding='utf-8') as f:
    f.write('*******************precision-recall-curve****************\n')
    f.write('*********************************************************\n')
    f.write('*********************************************************\n')

    f.write('Script run at ' + time.strftime('%Y-%m-%d %H:%M:%S\n'))
    f.write('dataset: nuswide\n')
    f.write('train on 5000 instances, hash bit length ' + str(hash_len) + '\n')

query_list, retrieval_list, training_list = utils.sample(query_num=2000, training_num=5000, dataset_size=20015)
training_text = utils.sample_feature(training_list, TEXT_FEATURE, modal='bow')
training_image = utils.sample_feature(training_list, IMAGE_FEATURE)

query_image = utils.sample_feature(query_list, IMAGE_FEATURE)
retrieval_image = utils.sample_feature(retrieval_list, IMAGE_FEATURE)
query_text = utils.sample_feature(query_list, TEXT_FEATURE, modal='bow')
retrieval_text = utils.sample_feature(retrieval_list, TEXT_FEATURE, modal='bow')
simi_matrix = utils.generate_similarity_matrix(query_list, retrieval_list, LABEL_PATH)

lssh = IMH(hash_len, training_image, training_text, query_image, query_text, retrieval_image, retrieval_text)
lssh.train()
lssh.eval(simi_matrix)