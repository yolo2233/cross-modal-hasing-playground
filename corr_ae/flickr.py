import numpy as np
import os
import re
import random
from PIL import Image
import tensorflow as tf
import gensim
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


PRUNE_TAGS = '../../dataset/flickr25k/organized_data/original_data/prune_tags_after_word2vec'
RESIZED_IMAGE = '../../dataset/flickr25k/organized_data/original_data/resized_pic'
SAVE_PATH = '../../dataset/flickr25k/organized_data'
WORD2VEC = '../../dataset/word2vec/GoogleNews-vectors-negative300.bin'
IMAGE_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_image_vgg19.npy'
TEXT_FEATURE = '../../dataset/flickr25k/organized_data/preprocess_data/flickr25k_text_bow.npy'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def generate_image_raw_feature(path=RESIZED_IMAGE):
    images = os.listdir(path)
    images.sort(key=lambda x: int(re.match(r'(im)(\d*)(.jpg)', x).group(2)))
    abs_image_path = [os.path.join(path, x) for x in images]

    image_raw_feature = np.asarray([np.asarray(Image.open(abs_path)) for abs_path in abs_image_path], dtype=np.float32)
    image_raw_feature /= 255

    images = [re.match(r'(im)(\d*)(.jpg)', x).group(2) for x in images]
    feature_dict = {}
    for index, image in enumerate(images):
        feature_dict.update({image: image_raw_feature[index, ...]})

    with open('../../dataset/flickr25k/organized_data/preprocess_data/image_numpy.dict', 'wb') as f:
        pickle.dump(feature_dict, f)


def generate_image_vgg16_feature(path=RESIZED_IMAGE):
    images = os.listdir(path)
    images.sort(key=lambda x: int(re.match(r'(im)(\d*)(.jpg)', x).group(2)))
    abs_image_path = [os.path.join(path, x) for x in images]

    image_raw_feature = np.asarray([np.asarray(Image.open(abs_path)) for abs_path in abs_image_path], dtype=np.float32)
    image_raw_feature[:, :, :, 0] -= 103.939
    image_raw_feature[:, :, :, 1] -= 116.779
    image_raw_feature[:, :, :, 2] -= 123.68

    vgg16_model = tf.keras.applications.VGG16(include_top=False)
    image_feature = vgg16_model.predict(image_raw_feature)
    image_feature = image_feature.reshape((image_raw_feature.shape[0], 25088))
    print(image_feature.shape)

    del image_raw_feature
    images = [re.match(r'(im)(\d*)(.jpg)', x).group(2) for x in images]

    feature_dict = {}
    for index, image in enumerate(images):
        feature_dict.update({image: image_feature[index, ...]})

    with open('../../dataset/flickr25k/organized_data/preprocess_data/image_conv.dict', 'wb') as f:
        pickle.dump(feature_dict, f)


def generate_image_vgg19_feature(path=RESIZED_IMAGE):
    images = os.listdir(path)
    images.sort(key=lambda x: int(re.match(r'(im)(\d*)(.jpg)', x).group(2)))
    abs_image_path = [os.path.join(path, x) for x in images]

    image_raw_feature = np.asarray([np.asarray(Image.open(abs_path)) for abs_path in abs_image_path], dtype=np.float32)
    image_raw_feature[:, :, :, 0] -= 103.939
    image_raw_feature[:, :, :, 1] -= 116.779
    image_raw_feature[:, :, :, 2] -= 123.68

    vgg19_model = tf.keras.applications.VGG19()
    extract_feature_model = tf.keras.models.Model(inputs=vgg19_model.input, outputs=vgg19_model.get_layer('fc1').output)
    image_feature = extract_feature_model.predict(image_raw_feature)

    print(image_feature.shape)
    images = [re.match(r'(im)(\d*)(.jpg)', x).group(2) for x in images]

    feature_dict = {}
    for index, image in enumerate(images):
        feature_dict.update({image: image_feature[index, ...]})

    with open('../../dataset/flickr25k/organized_data/preprocess_data/image_vgg19.dict', 'wb') as f:
        pickle.dump(feature_dict, f)


def generate_image_vgg_predict_feature(path=RESIZED_IMAGE):
    images = os.listdir(path)
    images.sort(key=lambda x: int(re.match(r'(im)(\d*)(.jpg)', x).group(2)))
    abs_image_path = [os.path.join(path, x) for x in images]

    image_raw_feature = np.asarray([np.asarray(Image.open(abs_path)) for abs_path in abs_image_path], dtype=np.float32)
    image_raw_feature[:, :, :, 0] -= 103.939
    image_raw_feature[:, :, :, 1] -= 116.779
    image_raw_feature[:, :, :, 2] -= 123.68

    vgg16_model = tf.keras.applications.VGG16()
    image_feature = vgg16_model.predict(image_raw_feature)

    images = [re.match(r'(im)(\d*)(.jpg)', x).group(2) for x in images]

    feature_dict = {}
    for index, image in enumerate(images):
        feature_dict.update({image: image_feature[index, ...]})

    with open('../../dataset/flickr25k/organized_data/preprocess_data/image_predict.dict', 'wb') as f:
        pickle.dump(feature_dict, f)


def generate_text_feature(path=PRUNE_TAGS):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)

    tags = os.listdir(path)
    tags.sort(key=lambda x: int(re.match(r'(tags)(\d*)(.txt)', x).group(2)))

    tag_abs_path = [os.path.join(path, tag) for tag in tags]
    feature_dict = {}

    for abs_path in tag_abs_path:
        with open(abs_path, encoding='utf-8') as f:
            b = [i[:-1] for i in f.readlines()]
            invalid_words = 0
            word_embedding = 0
            for i in b:
                try:
                    word_embedding += word2vec[i]
                except KeyError:
                    invalid_words += 1
            word_embedding /= (len(b) - invalid_words)
            feature_dict.update({re.match(r'(tags)(\d*)(.txt)', os.path.split(abs_path)[1]).group(2): word_embedding})

    with open('../../dataset/flickr25k/organized_data/preprocess_data/text_word2vec.dict', 'wb') as f:
        pickle.dump(feature_dict, f)


if __name__ == '__main__':
    pass
