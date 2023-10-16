# coding=utf-8
# Yujay@2023/10/16

import tensorflow as tf
from reader import get_samples
from models import DNN, DynamicDNN

if __name__ == "__main__":
    # model = DNN()                                 # baseline
    # model = DynamicDNN("lookup", "fc", "add")     # v1
    # model = DynamicDNN("lookup", "fc", "fc")      # v2
    model = DynamicDNN("lookup", "lora", "add")   # v3
    # model = DynamicDNN("lookup", "lora", "fc")    # v4
    # model = DynamicDNN("mlp", "fc", "add")        # v5
    # model = DynamicDNN("mlp", "fc", "fc")         # v6
    # model = DynamicDNN("mlp", "lora", "add")      # v7
    # model = DynamicDNN("mlp", "lora", "fc")       # v8
    model.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        writer = tf.summary.FileWriter("logs", sess.graph)

        for iter in range(10):
            one_batch = get_samples()
            _, loss, auc = sess.run(
                [model.train_op, model.loss, model.auc],
                feed_dict={
                    model.is_training: True,
                    model.inputs["labels"]: one_batch["labels"],
                    model.inputs["f1"]: one_batch["features"][0],
                    model.inputs["f2"]: one_batch["features"][1],
                    model.inputs["f3"]: one_batch["features"][2],
                    model.inputs["f4"]: one_batch["features"][3],
                    model.inputs["f5"]: one_batch["features"][4],
                    model.inputs["domain_indicators"]: one_batch["domain_indicators"],
                }
            )
            print "loss:{}, auc:{}".format(loss, auc[0])
