from utils import *

_all__ = ['image_decoder', 'image_encoder', 'text_encoder', 'text_decoder']


def image_encoder(input_, neuron_nums, keep_prob_ph, training):
    en_fc1 = fully_connected(input_, neuron_nums[0], 'image_en_fc1', keep_prob=keep_prob_ph, training=training)
    en_fc2 = fully_connected(en_fc1, neuron_nums[1], 'image_en_fc2', keep_prob=keep_prob_ph, training=training)

    return en_fc2


def image_decoder(input_, neuron_nums, keep_prob_ph, training):
    de_fc1 = fully_connected(input_, neuron_nums[0], 'image_de_fc1', keep_prob=keep_prob_ph, training=training)
    de_fc2 = fully_connected(de_fc1, neuron_nums[1], 'image_de_fc2', batch_norm=False)

    return de_fc2


def text_encoder(input_, neuron_nums, keep_prob_ph, training):
    en_fc1 = fully_connected(input_, neuron_nums[0], 'text_en_fc1', keep_prob=keep_prob_ph, training=training)
    en_fc2 = fully_connected(en_fc1, neuron_nums[1], 'text_en_fc2', keep_prob=keep_prob_ph, training=training)

    return en_fc2


def text_decoder(input_, neuron_nums, keep_prob_ph, training):
    de_fc1 = fully_connected(input_, neuron_nums[0], 'text_de_fc1', keep_prob=keep_prob_ph, training=training)
    de_fc2 = fully_connected(de_fc1, neuron_nums[1], 'text_de_fc2', batch_norm=False)

    return de_fc2