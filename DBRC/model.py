import tensorflow as tf
from utils import optimized_mAP, precision_top_k, count_time, precision_recall
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from ops import *
from utils import fully_connected

class DBRC(object):

    def __init__(self, input_len_img, input_len_txt, latent_len):
        # data:
        self.latent_len = latent_len

        self.loss_summaries = []

        self.data_ph = tf.placeholder(tf.float32, shape=[None, input_len_img + input_len_txt])
        self.image_ph = tf.placeholder(tf.float32, shape=[None, input_len_img])
        self.text_ph = tf.placeholder(tf.float32, shape=[None, input_len_txt])
        self.keep_prob_ph = tf.placeholder(tf.float32)
        self.training_ph = tf.placeholder(tf.bool)
        self.alpha = tf.get_variable('alpha', shape=[1, self.latent_len], dtype=tf.float32, initializer=tf.initializers.constant(1.0))
        self.train_mAP = None

        self.input_len_img = input_len_img
        self.input_len_txt = input_len_txt

        self.summary_writer = None

        self.image_latent = None
        self.image_tilde = None

        self.text_latent = None
        self.text_tilde = None
        self.fuse_hashcode = None
        self.latent_loss = None
        self.recon_image_loss = None
        self.recon_text_loss = None

    def build_graph(self):
        self.image_latent = image_encoder(self.image_ph, [128, 512], self.keep_prob_ph,
                                          training=self.training_ph)

        self.text_latent = text_encoder(self.text_ph, [128, 512], self.keep_prob_ph,
                                        training=self.training_ph)

        # atanh
        fuse_latent = tf.concat([self.image_latent, self.text_latent], axis=1)
        dense_latent = fully_connected(fuse_latent, 512, 'dense_latent')
        coding_layer = fully_connected(dense_latent, self.latent_len, 'coding_layer', activation=None)

        self.fuse_hashcode = tf.tanh(self.alpha * coding_layer) + 0.001 * tf.norm((1 / self.alpha)) ** 2

        self.image_tilde = image_decoder(self.fuse_hashcode, [128, self.input_len_img], self.keep_prob_ph,
                                         training=self.training_ph)

        self.text_tilde = text_decoder(self.fuse_hashcode, [128, self.input_len_txt], self.keep_prob_ph,
                                       training=self.training_ph)

        self.recon_image_loss = tf.reduce_mean(tf.square(self.image_ph - self.image_tilde))
        self.recon_text_loss = tf.reduce_mean(tf.square(self.text_ph - self.text_tilde))

        self._classify_vars()
        self._init_summary()

    def train(self, train_feed_dict: dict, test_feed_dict: dict, result_file):

        recon_loss = self.recon_image_loss + self.recon_text_loss

        loss = recon_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            loss_opt = tf.train.RMSPropOptimizer(train_feed_dict['lr']).minimize(loss)

        dataset = tf.data.Dataset.from_tensor_slices(self.data_ph).repeat().batch(train_feed_dict['batch_size'])
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        self.summary_writer = tf.summary.FileWriter(train_feed_dict['snapshot'], graph=tf.get_default_graph())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={self.data_ph: train_feed_dict['data']})
            # saver = tf.train.Saver()
            for i in range(train_feed_dict['iteration']):

                image_element, text_element = tf.split(next_element, [self.input_len_img, self.input_len_txt], axis=1)
                image_batch, text_batch = sess.run([image_element, text_element])

                feed_dict = {self.image_ph: image_batch, self.text_ph: text_batch,
                             self.keep_prob_ph: train_feed_dict['keep_prob'], self.training_ph: True}

                _, image_loss_, text_loss_ = sess.run(
                    [loss_opt, self.recon_image_loss, self.recon_text_loss], feed_dict=feed_dict)

                summaries_val = sess.run(self.loss_summaries, feed_dict=feed_dict)

                if i % 10 == 0:
                    self._add_summary(summaries_val, i)
                if i % 50 == 0:
                    print('At Iteration %d, recon image loss is %.5f, recon text loss is %.5f' % (i, image_loss_, text_loss_))

                if i % 500 == 0 and i > 0:
                    image_query = sess.run(self.fuse_hashcode,
                                           feed_dict={self.image_ph: test_feed_dict['image_query'],
                                                      self.text_ph: np.zeros((test_feed_dict['image_query'].shape[0], self.input_len_txt)),
                                                      self.keep_prob_ph: 1.0,
                                                      self.training_ph: False})

                    text_query = sess.run(self.fuse_hashcode,
                                          feed_dict={self.text_ph: test_feed_dict['text_query'],
                                                     self.image_ph: np.zeros((test_feed_dict['text_query'].shape[0], self.input_len_img)),
                                                     self.keep_prob_ph: 1.0,
                                                     self.training_ph: False})

                    image_re = sess.run(self.fuse_hashcode,
                                        feed_dict={self.image_ph: test_feed_dict['image_re'],
                                                   self.text_ph: np.zeros((test_feed_dict['image_re'].shape[0], self.input_len_txt)),
                                                   self.keep_prob_ph: 1.0,
                                                   self.training_ph: False})

                    text_re = sess.run(self.fuse_hashcode,
                                       feed_dict={self.text_ph: test_feed_dict['text_re'],
                                                  self.image_ph: np.zeros((test_feed_dict['text_re'].shape[0], self.input_len_img)),
                                                  self.keep_prob_ph: 1.0,
                                                  self.training_ph: False})

                    # i2t
                    hash_MAP_i2t = optimized_mAP(image_query, text_re, test_feed_dict['similarity_matrix'],
                                                 dis_metric=test_feed_dict['metric'])
                    #
                    hash_top_k_precision_i2t = precision_top_k(image_query, text_re,
                                                               test_feed_dict['similarity_matrix'],
                                                               [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                                                                600, 650, 700, 750, 800, 850, 900, 950, 1000],
                                                               test_feed_dict['metric'])

                    # precision_i2t, recall_i2t = precision_recall(image_query, text_re, test_feed_dict['similarity_matrix'])

                    with open(result_file, 'a') as f:
                        f.write('At iteration ' + str(i) + '\n')
                        f.write('i2t hash MAP  is %.5f\n' % hash_MAP_i2t)
                        f.write(
                            'i2t top [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] hash precision is: ' + str(
                                hash_top_k_precision_i2t) + '\n')
                        # f.write('i2t precision: ' + str(precision_i2t) + '\n')
                        # f.write('t2i recall: ' + str(recall_i2t) + '\n')

                    # t2i
                    hash_MAP_t2i = optimized_mAP(text_query, image_re, test_feed_dict['similarity_matrix'],
                                                 dis_metric=test_feed_dict['metric'])
                    #
                    hash_top_k_precision_t2i = precision_top_k(text_query, image_re,
                                                               test_feed_dict['similarity_matrix'],
                                                               [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                                                                600, 650, 700, 750, 800, 850, 900, 950, 1000],
                                                               test_feed_dict['metric'])

                    # precision_i2t, recall_t2i = precision_recall(text_query, image_re, test_feed_dict['similarity_matrix'])

                    with open(result_file, 'a') as f:
                        f.write('t2i hash MAP is %.5f\n' % hash_MAP_t2i)
                        f.write(
                            't2i top [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] precision is: ' + str(
                                hash_top_k_precision_t2i) + '\n\n')
                        # f.write('t2i precision: ' + str(precision_i2t) + '\n')
                        # f.write('t2i recall: ' + str(recall_t2i) + '\n\n')

            # saver.save(sess, 'saved_model/coco_model_32.ckpt')

    @count_time
    def eval(self, query, retrieval, similarity_matrix, query_modal, retrieval_modal, dis_metric, radius=None):
        saver = tf.train.Saver(var_list=self.image_encoder_vars + self.text_encoder_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, 'saved_model/model.ckpt')

            if query_modal == 'img':
                query_latent = sess.run(self.image_latent,
                                        feed_dict={self.image_ph: query, self.keep_prob_ph: 1.0,
                                                   self.training_ph: False})
            else:
                query_latent = sess.run(self.text_latent,
                                        feed_dict={self.text_ph: query, self.keep_prob_ph: 1.0,
                                                   self.training_ph: False})

            if retrieval_modal == 'img':
                retrieval_latent = sess.run(self.image_latent,
                                            feed_dict={self.image_ph: retrieval, self.keep_prob_ph: 1.0,
                                                       self.training_ph: False})
            else:
                retrieval_latent = sess.run(self.text_latent,
                                            feed_dict={self.text_ph: retrieval, self.keep_prob_ph: 1.0,
                                                       self.training_ph: False})

        MAP = optimized_mAP(query_latent, retrieval_latent, similarity_matrix, dis_metric=dis_metric)
        top_k_precision = precision_top_k(query_latent, retrieval_latent, similarity_matrix, [10, 20, 50, 100, 500],
                                          dis_metric)

        if dis_metric == 'hash':
            MAP_comp = optimized_mAP(query_latent, retrieval_latent, similarity_matrix, dis_metric='cosine')
            top_k_precision_comp = precision_top_k(query_latent, retrieval_latent, similarity_matrix,
                                                   [10, 20, 50, 100, 500], dis_metric='cosine')
            precision, recall = precision_recall(query_latent, retrieval_latent, similarity_matrix)

            return MAP, top_k_precision, MAP_comp, top_k_precision_comp, precision, recall

        return MAP, top_k_precision

    def _init_summary(self):
        # visualization on tensorboard
        self.recon_image_loss_sum = tf.summary.scalar('recon_image_loss', self.recon_image_loss)
        self.recon_text_loss_sum = tf.summary.scalar('recon_text_loss', self.recon_text_loss)

        self.loss_summaries.extend([self.recon_image_loss_sum,
                                    self.recon_text_loss_sum])

    def _add_summary(self, summaries, it):
        for summary in summaries:
            self.summary_writer.add_summary(summary, it)

    def _classify_vars(self):
        vars = tf.global_variables()
        self.image_encoder_vars = [var for var in vars if 'image_en' in var.name]
        self.image_decoder_vars = [var for var in vars if 'image_de' in var.name]
        self.text_encoder_vars = [var for var in vars if 'text_en' in var.name]
        self.text_decoder_vars = [var for var in vars if 'text_de' in var.name]


if __name__ == '__main__':
    pass
