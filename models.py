# -*- coding:utf-8 -*-
# Yujay@2023/10/16
import tensorflow as tf
from const import *
from dynamic_layers import dynamic_dense

class DNN(object):
    def __init__(self):
        self.inputs = {}
        self.tables = {}
        self.features = {}

        with tf.name_scope("input"):
            self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")
            self.inputs["labels"] = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="labels")
            for index in range(FEATURE_NUM):
                key = "f{}".format(index+1)
                self.inputs[key] = tf.placeholder(dtype=tf.int32, shape=[None,], name=key)
            self.inputs["domain_indicators"] = tf.placeholder(dtype=tf.int32, shape=[None,], name="domain_indicators")

        with tf.variable_scope("embedding_tables"):
            for index in range(FEATURE_NUM):
                key = "f{}".format(index+1)
                self.tables[key] = tf.get_variable(name=key, shape=[FEATURE_BUCKET_SIZE, EMBEDDING_SIZE])

    def build_embedding_layer(self):
        with tf.variable_scope("build_embedding_layer"):
            for index in range(FEATURE_NUM):
                key = "f{}".format(index+1)
                self.features[key] = tf.nn.embedding_lookup(
                    self.tables[key],
                    self.inputs[key],
                    name=key
                )

    def build_main_net(self):
        with tf.variable_scope("build_main_net", reuse=tf.AUTO_REUSE):
            feats = []
            for index in range(FEATURE_NUM):
                key = "f{}".format(index+1)
                feats.append(self.features[key])
            feat = tf.concat(feats, axis=-1)

            for index, hidden_units in enumerate(HIDDEN_UNITS):
                feat = tf.layers.dense(
                    inputs=feat,
                    units=hidden_units,
                    activation=tf.nn.relu,
                    use_bias=True,
                    name="hidden_layer_{}".format(index+1)
                )
                feat = tf.layers.batch_normalization(
                    inputs=feat,
                    training=self.is_training,
                    name="batch_norm_{}".format(index+1)
                )

            self.logits = tf.layers.dense(
                inputs=feat,
                units=1,
                activation=None,
                name="logits"
            )
            self.pred = tf.sigmoid(self.logits)

    def build_loss(self):
        with tf.variable_scope("build_loss"):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.inputs["labels"]
            )
            self.loss = tf.reduce_mean(loss)

    def build_train_op(self):
        with tf.variable_scope("build_train_op"):
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=LEARNING_RATE
            ).minimize(self.loss)

    def build_metrics(self):
        with tf.variable_scope("build_metrics"):
            self.auc = tf.metrics.auc(
                labels=self.inputs["labels"],
                predictions=self.pred,
            )

    def build(self):
        self.build_embedding_layer()
        self.build_main_net()
        self.build_loss()
        self.build_train_op()
        self.build_metrics()

class DynamicDNN(DNN):
    def __init__(self,
                 dam="lookup",
                 ddm="lora",
                 dim="add",
                 rank=8):
        super(DynamicDNN, self).__init__()
        self.dam = dam
        self.ddm = ddm
        self.dim = dim
        self.rank = rank

    def build_main_net(self):
        with tf.variable_scope("build_main_net", reuse=tf.AUTO_REUSE):
            feats = []
            for index in range(FEATURE_NUM):
                key = "f{}".format(index+1)
                feats.append(self.features[key])
            feat = tf.concat(feats, axis=-1)
            input_dim = FEATURE_NUM*EMBEDDING_SIZE

            for index, hidden_units in enumerate(HIDDEN_UNITS):
                feat = dynamic_dense(
                    inputs=feat,
                    domain_indicators=self.inputs["domain_indicators"],
                    input_dim=input_dim,
                    output_dim=hidden_units,
                    activation=tf.nn.relu,
                    name="hidden_layer_{}".format(index+1),
                    dam=self.dam,
                    ddm=self.ddm,
                    dim=self.dim,
                    rank=self.rank,
                    is_training=self.is_training
                )
                feat = tf.layers.batch_normalization(
                    inputs=feat,
                    training=self.is_training,
                    name="batch_norm_{}".format(index+1)
                )
                input_dim = hidden_units

            self.logits = tf.layers.dense(
                inputs=feat,
                units=1,
                activation=None,
                name="logits"
            )
            self.pred = tf.sigmoid(self.logits)

if __name__ == '__main__':
    pass