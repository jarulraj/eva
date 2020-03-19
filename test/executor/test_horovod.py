# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import os
import subprocess
from distutils.version import LooseVersion

import numpy as np

import pyspark.sql.types as T
from pyspark import SparkConf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import horovod.spark.keras as hvd
from horovod.spark.common.store import Store


class HorovodTest(unittest.TestCase):

    def test_horovod(self):

        work_dir = '/tmp'
        data_dir = '/tmp'
        epochs = 1
        batch_size = 128
        num_proc = 2

        # Initialize SparkSession
        conf = SparkConf().setAppName('keras_spark_mnist').set(
            'spark.sql.shuffle.partitions', '16')

        conf.setMaster('local[{}]'.format(num_proc))
        spark = SparkSession.builder.config(conf=conf).getOrCreate()

        # Setup our store for intermediate data
        store = Store.create(work_dir)

        # Download MNIST dataset
        data_url = \
            'https://www.csie.ntu.edu.tw/' +\
            '~cjlin/libsvmtools/datasets/multiclass/mnist.bz2'
        libsvm_path = os.path.join(data_dir, 'mnist.bz2')
        if not os.path.exists(libsvm_path):
            subprocess.check_output(['wget', data_url, '-O', libsvm_path])

        # Load dataset into a Spark DataFrame
        df = spark.read.format('libsvm') \
            .option('numFeatures', '784') \
            .load(libsvm_path)

        # One-hot encode labels into SparseVectors
        encoder = OneHotEncoderEstimator(inputCols=['label'],
                                         outputCols=['label_vec'],
                                         dropLast=False)
        model = encoder.fit(df)
        train_df = model.transform(df)

        # Train/test split
        train_df, test_df = train_df.randomSplit([0.9, 0.1])

        # Disable GPUs when building the model to prevent memory leaks
        if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
            # See https://github.com/tensorflow/tensorflow/issues/33168
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            keras.backend.set_session(
                tf.Session(
                    config=tf.ConfigProto(
                        device_count={
                            'GPU': 0})))

        # Define the Keras model without any Horovod-specific parameters
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        optimizer = keras.optimizers.Adadelta(1.0)
        loss = keras.losses.categorical_crossentropy

        # Train a Horovod Spark Estimator on the DataFrame
        keras_estimator = hvd.KerasEstimator(num_proc=num_proc,
                                             store=store,
                                             model=model,
                                             optimizer=optimizer,
                                             loss=loss,
                                             metrics=['accuracy'],
                                             feature_cols=['features'],
                                             label_cols=['label_vec'],
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             verbose=1)

        keras_model = keras_estimator.fit(
            train_df).setOutputCols(['label_prob'])

        # Evaluate the model on the held-out test DataFrame
        pred_df = keras_model.transform(test_df)
        argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
        pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
        evaluator = MulticlassClassificationEvaluator(
            predictionCol='label_pred', 
            labelCol='label', 
            metricName='accuracy')
        print('Test accuracy:', evaluator.evaluate(pred_df))

        spark.stop()
