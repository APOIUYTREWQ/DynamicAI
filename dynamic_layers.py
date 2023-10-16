# -*- coding:utf-8 -*-
# Yujay@2023/10/16
import tensorflow as tf
from const import *


def dynamic_weights_mlp(domain_indicators, input_dim, output_dim, is_training, name):
    table = tf.get_variable(name=name + "_domain_indicators", shape=[DOMAIN_BUCKET_SIZE, 8])
    domain_input = tf.nn.embedding_lookup(table, domain_indicators)
    for index, hidden_units in enumerate([128, 32, 8]):
        domain_input = tf.layers.dense(
            inputs=domain_input,
            units=hidden_units,
            activation=tf.nn.relu,
            use_bias=True,
            name=name + "_dam_{}".format(index + 1)
        )
        domain_input = tf.layers.batch_normalization(
            inputs=domain_input,
            training=is_training,
            name=name + "_bn_{}".format(index + 1)
        )
    weights = tf.layers.dense(
        inputs=domain_input,
        units=input_dim * output_dim,
        activation=None,
        use_bias=True,
        name=name + "_dam_linear"
    )
    weights = tf.reshape(weights, [-1, input_dim, output_dim])
    return weights

def dynamic_weights_lookup(domain_indicators, input_dim, output_dim, is_training, name):
    table = tf.get_variable(name=name + "_matrix", shape=[DOMAIN_BUCKET_SIZE, input_dim * output_dim])
    weights = tf.nn.embedding_lookup(table, domain_indicators)
    weights = tf.reshape(weights, [-1, input_dim, output_dim])
    return weights

def dynamic_dense(inputs,
                  domain_indicators,
                  input_dim,
                  output_dim,
                  activation=None,
                  use_bias=True,
                  dam="lookup",
                  ddm="lora",
                  dim="add",
                  rank=8,
                  is_training=True,
                  name=""):
    print "dynamic_dense: name={}, dam={}, ddm={}, dim={}".format(name, dam, ddm, dim)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        static_feat = tf.layers.dense(
            inputs=inputs,
            units=output_dim,
            activation=None,
            use_bias=use_bias,
            name=name + "_static_dense"
        )

        feat = tf.expand_dims(inputs, axis=1)

        if ddm == "lora":
            if dam == "lookup":
                weights_a = dynamic_weights_lookup(
                    domain_indicators=domain_indicators,
                    input_dim=input_dim,
                    output_dim=rank,
                    is_training=is_training,
                    name=name + "_weights_a"
                )
                weights_b = dynamic_weights_lookup(
                    domain_indicators=domain_indicators,
                    input_dim=rank,
                    output_dim=output_dim,
                    is_training=is_training,
                    name=name + "_weights_b"
                )
                if use_bias:
                    bias = dynamic_weights_lookup(
                        domain_indicators=domain_indicators,
                        input_dim=1,
                        output_dim=output_dim,
                        is_training=is_training,
                        name=name + "_bias"
                    )
            elif dam == "mlp":
                weights_a = dynamic_weights_mlp(
                    domain_indicators=domain_indicators,
                    input_dim=input_dim,
                    output_dim=rank,
                    is_training=is_training,
                    name=name + "_weights_a"
                )
                weights_b = dynamic_weights_mlp(
                    domain_indicators=domain_indicators,
                    input_dim=rank,
                    output_dim=output_dim,
                    is_training=is_training,
                    name=name + "_weights_b"
                )
                if use_bias:
                    bias = dynamic_weights_mlp(
                        domain_indicators=domain_indicators,
                        input_dim=1,
                        output_dim=output_dim,
                        is_training=is_training,
                        name=name + "_bias"
                    )
            feat = tf.matmul(feat, weights_a)
            feat = tf.matmul(feat, weights_b)
        elif ddm == "fc":
            if dam == "lookup":
                weights = dynamic_weights_lookup(
                    domain_indicators=domain_indicators,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    is_training=is_training,
                    name=name + "_weights"
                )
                if use_bias:
                    bias = dynamic_weights_lookup(
                        domain_indicators=domain_indicators,
                        input_dim=1,
                        output_dim=output_dim,
                        is_training=is_training,
                        name=name + "_bias"
                    )
            if dam == "mlp":
                weights = dynamic_weights_mlp(
                    domain_indicators=domain_indicators,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    is_training=is_training,
                    name=name + "_weights"
                )
                if use_bias:
                    bias = dynamic_weights_mlp(
                        domain_indicators=domain_indicators,
                        input_dim=1,
                        output_dim=output_dim,
                        is_training=is_training,
                        name=name + "_bias"
                    )
            feat = tf.matmul(feat, weights)

        if use_bias:
            feat = feat + bias
        dynamic_feat = tf.reshape(feat, [-1, output_dim])

        if dim == "add":
            outputs = static_feat + dynamic_feat
        if dim == "fc":
            feat = tf.concat([static_feat, dynamic_feat], axis=-1)
            outputs = tf.layers.dense(
                inputs=feat,
                units=output_dim,
                activation=None,
                use_bias=use_bias,
                name=name + "_integration"
            )

        if activation is not None:
            outputs = activation(outputs)

        return outputs


if __name__ == '__main__':
    pass