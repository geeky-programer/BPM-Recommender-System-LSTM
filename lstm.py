## LSTM model build and train
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dropout,
    Dense,
    Activation,
    Concatenate,
    Lambda,
)
from tensorflow.keras.models import load_model, Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
#from utils import Predictor, Prediction, Type
from utils import get_label_mapping, get_labeled_paths, append_path, build_request_trace
from utils import remove_active_element, cutoff_behind_active_element
import time

class Lstm():

    def create(
        self,
        vocabulary,
        hidden_size,
        hidden_size_neighbor,
        use_dropout,
        neighbor_count
    ):
        input_element = Input((None,))
        l1_1 = Embedding(len(vocabulary), hidden_size)(input_element)

        input_predecessors = Input((None, neighbor_count))
        input_successors = Input((None, neighbor_count))
        if use_dropout:
            l2_0_d = Embedding(len(vocabulary), hidden_size_neighbor)(
                input_predecessors
            )
            l2_1 = Dropout(0.7)(l2_0_d)

            l3_0_d = Embedding(len(vocabulary), hidden_size_neighbor)(input_successors)
            l3_1 = Dropout(0.7)(l3_0_d)
        else:
            l2_1 = Embedding(len(vocabulary), hidden_size_neighbor)(input_predecessors)

            l3_1 = Embedding(len(vocabulary), hidden_size_neighbor)(input_successors)

        # flatten last to dimensions
        def squeeze_last2_operator(x4d):
            shape = tf.shape(x4d)  # get dynamic tensor shape
            x3d = tf.reshape(
                x4d, [shape[0], shape[1], hidden_size_neighbor * neighbor_count]
            )
            return x3d

        def squeeze_last2_shape(x4d_shape):
            in_batch, in_rows, in_cols, in_filters = x4d_shape
            if None in [in_cols, in_filters]:
                output_shape = (
                    in_batch,
                    in_rows,
                    hidden_size_neighbor * neighbor_count,
                )
            else:
                output_shape = (
                    in_batch,
                    in_rows,
                    hidden_size_neighbor * neighbor_count,
                )
            return output_shape

        l2_2 = Lambda(squeeze_last2_operator, output_shape=squeeze_last2_shape)(l2_1)
        l2_3 = Dense(hidden_size)(l2_2)

        l3_2 = Lambda(squeeze_last2_operator, output_shape=squeeze_last2_shape)(l3_1)
        l3_3 = Dense(hidden_size)(l3_2)

        lc_1 = Concatenate(axis=-1)([l1_1, l2_3, l3_3])
        lc_2 = LSTM(hidden_size, return_sequences=True)(lc_1)
        lc_3 = LSTM(hidden_size, return_sequences=True)(lc_2)
        if use_dropout:
            lc_3_d = Dropout(0.2)(lc_3)
            lc_4 = tf.keras.layers.TimeDistributed(Dense(len(vocabulary)))(lc_3_d)
        else:
            lc_4 = tf.keras.layers.TimeDistributed(Dense(len(vocabulary)))(lc_3)
        out = Activation("softmax")(lc_4)

        self.model = Model(
            inputs=[input_element, input_predecessors, input_successors], outputs=out
        )
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["categorical_accuracy"],
        )

    def load(self):
        self.model = load_model(self.path + self.name)
        self.model.make_predict_function()
        self.available = True

    def train(
        self,
        train_data_generator,
        valid_data_generator,
        train_data_length,
        valid_data_length,
        num_epochs
    ):
        history = self.model.fit(
            x=train_data_generator.generate(),
            steps_per_epoch=train_data_length,
            epochs=num_epochs,
            validation_data=valid_data_generator.generate(),
            validation_steps=valid_data_length,
        )

        self.model.save(self.path + self.name)
        self.available = True
        # Plot training & validation accuracy values
        # print("History :{}".format(history.history))
        plt.plot(history.history["categorical_accuracy"])
        # plt.plot(history.history['val_categorical_accuracy'])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Training"], loc="upper left")
        plt.ylim([0, 1])
        plt.savefig(self.path + self.name + "_train_acc.png")
        plt.clf()

        # Plot training & validation loss values
        plt.plot(history.history["loss"])
        # plt.plot(history.history['val_loss'])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Training"], loc="upper left")
        plt.ylim([0, None])
        plt.savefig(self.path + self.name + "_train_loss.png")

        return max(history.history["categorical_accuracy"]), min(
            history.history["loss"]
        )

    def predict_trace(self, data):
        if self.available:
            formatted_data = [
                np.zeros((1, len(data[0]))),
                np.zeros((1, len(data[0]), 10)),
                np.zeros((1, len(data[0]), 10)),
            ]
            formatted_data[0][0, :] = data[0]
            for i, p in enumerate(data[1]):
                if len(p) > 0:
                    for ii, pp in enumerate(p):
                        formatted_data[1][0, i, ii] = pp
            for i, p in enumerate(data[2]):
                if len(p) > 0:
                    for ii, pp in enumerate(p):
                        formatted_data[2][0, i, ii] = pp
            prediction = self.model.predict(formatted_data)
            predict_elements = []
            for pe in prediction[0]:
                predict_elements.append(np.argmax(pe))
            return predict_elements
        return None
