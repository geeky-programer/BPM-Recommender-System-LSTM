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
from ...utility.models import Predictor, Prediction, Type
from ...utility.mapping_reader import get_label_mapping, get_labeled_paths, append_path, build_request_trace
from ...utility.trace_mutator import remove_active_element, cutoff_behind_active_element
from ...similarity.similaritypredictor import SimilarityPredictor
from ...similarity.similarity import SimilarityWrapper
from ...similarity.sentence import Sentence
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

    def predict(
        self, path: str, activeElement: str, excludeActiveElement: bool = True
    ) -> list[Prediction]:
        #Imp: "path" parameter here contains only ouput.json with information about the current modeling.

        # Create list of relevant traces and mapping = (id, {label,type})
        #dict_id_path = {0:[1,2,3],1:[3,4,5]}
        #dict_label_path = {0:['','',''],1:['','']}
        request_mapping  = get_label_mapping(path)
        dict_id_keys, dict_id_path, dict_label_path = get_labeled_paths()

        print("Request_mapping:{}".format(request_mapping))
        print('Total id  paths:{}'.format(len(dict_id_path)))
        print('Total labeled paths:{}'.format(len(dict_label_path)))

        with open(path) as f:
            result_json = json.load(f)

        print(result_json)

        print("Path:{}\n activeelement:{} ".format(path,activeElement))
        request_traces = []
        for _, trace_json in result_json["paths"].items():
            trace = []
            trace_labels = []
            # for model, model_paths in trace_json.items():
            for _, element in trace_json.items():
                trace.append(
                        (
                            element["id"],
                            element["predecessors"][0],
                            element["successors"][0],
                        )
                )
                trace_labels.append(request_mapping[element["id"]]['label'])
            # print("trace:{}".format(trace))
            # Cut off trace behind active element
            trace = cutoff_behind_active_element(trace, request_mapping, activeElement)
            if activeElement:
                trace_labels = trace_labels[:-1]

            if activeElement in request_mapping[trace[-1][0]]["bpmnId"]:

                # Check if we need to exclude the last trace element
                if excludeActiveElement:
                    trace = remove_active_element(trace)

                request_traces.append(trace)
        print("Request traces:{}".format(request_traces))

        #Collect semantically similar sentences and get the paths
        #Note : request_mapping contains just the mapping of id to the label and type
        #Step1: Extract labels first
        # label_path = []
        # for path in trace:
        #     node_id = path[0]
        #     #Search for the label in the
        #     label_path.append(request_mapping[node_id]['label'])

        print("Label paths:{}".format(trace_labels))
        print("Current label:{}".format(trace_labels[-1]))

        start_time = time.time()
        #Got the label path, call similarity to get paths
        #Smart way - first get the matching label only for the lst ABC, XYC' -> Get C'
        sim = SimilarityPredictor(trace_labels[-1])
        matching_phrases = sim.get_matching_phrases()

        # print("Similar phrases:{}".format(matching_phrases))

        #Get labeled paths
        # dict_id_path, dict_label_path = get_labeled_paths(path)

        #Case 1: Get paths with only the last label similar ABC, EFC'

        # for i in range(len(matching_phrases)):
        #     matching_paths = []
        #     for j in range(len(labeled_paths)):
        #         if matching_phrases[i] in labeled_paths[j]:
        #             path = labeled_paths[j]
        #             idx_list = [idx + 1 for idx, val in enumerate(path) if val == matching_phrases[i]]
        #             res = [path[i: j] for i, j in zip([0] + idx_list, idx_list + ([len(path)] if idx_list[-1] != len(path) else []))]
        #             matching_paths.append(res[0])
        #     if len(matching_paths) > 0:
        #         matching_dict[i] = matching_paths
        # #n*n
        # print('Dict id path:{}'.format(dict_id_path))
        # print('Dict label path:{}'.format(dict_label_path))

        matching_dict = {}
        matching_id_dict = {}
        for i in range(len(matching_phrases)):
            matching_paths = []
            matching_id_paths = []
            index = 0
            for key, path in dict_label_path.items():
                if matching_phrases[i] in path:
                    idx_list = [idx+1 for idx, val in enumerate(path) if val == matching_phrases[i]]
                    # print('idx list:{}'.format(idx_list))
                    res = [path[i: j] for i, j in zip([0] + idx_list, idx_list + ([len(path)] if idx_list[-1] != len(path) else []))]
                    matching_paths.append(res[0])
                    # print(res[0])
                    #You have the path here
                    #Similarly cut the id path also
                    #get id path using key
                    id_p = dict_id_path[key]
                    id_path_res = [id_p[i: j] for i, j in zip([0] + idx_list, idx_list + ([len(id_p)] if idx_list[-1] != len(id_p) else []))]
                    matching_id_paths.append(id_path_res[0])
                    # print(id_path_res[0])
                    # print(key)
                    # print('Matching paths:{}'.format(matching_paths))
            if len(matching_paths) > 0:
                matching_dict[i] = matching_paths
                matching_id_dict[i] = matching_id_paths
                #Do the same for ids
                # id_path_from_dict = dict_id_path[key]
                # # print(id_path_from_dict)
                # sliced_path = id_path_from_dict[:len(matching_paths)]
                # # print('Checking length of label and id path = {}, {}'.format(len(matching_paths),len(sliced_path)))
                # dict_id_path[key] = sliced_path
        print('Matching label paths:{}'.format(len(matching_dict)))
        print('Matching id paths:{}'.format(len(matching_id_dict)))

        #Case 2: Get paths with combination of all the labels in the path similar ABC, A'B'C'
        #Case 2: Matching all path nodes
        incoming_path_appended = append_path(trace_labels)
        short_listed_paths = []
        short_listed_id_paths = []
        count = 0
        for key, label_array in matching_dict.items():
            #Append the the labels seperated by '/'
            for i in range(len(label_array)):
                if(label_array[i] not in short_listed_paths):
                    path_appended = append_path(label_array[i])
                    # Get similarity score for paths
                    # print('Sent1:{}'.format(incoming_path_appended))
                    # print('Sent2:{}'.format(path_appended))
                    sent1 = Sentence(incoming_path_appended)
                    sent2 = Sentence(path_appended)
                    wrp = SimilarityWrapper(sent1, sent2)
                    wrp.calculate_similarity_score()
                    count+=1
                    if wrp.overall_score > 0.75:#Setting higher threshold to keep paths more similar
                        print(f'{wrp.overall_score}, {label_array[i]}. {matching_id_dict[key][i]}')
                        short_listed_paths.append(label_array[i])
                        short_listed_id_paths.append(matching_id_dict[key][i])

        #Consider all of short_listed_paths for prediction
        for x,y  in zip(short_listed_paths,short_listed_id_paths):
            print("Shortlisted paths:{} \n id :{}\n".format(x,y))

        # print('Sl paths:{}'.format(short_listed_paths))
        print('total loops:{}'.format(count))
        print('Total sl paths:{}'.format(len(short_listed_paths)))

        shortlisted_request_trace = build_request_trace(short_listed_id_paths)

        print("Time taken for shortlisting paths: %s seconds ---" % (time.time() - start_time))

        #Build the request trace and append to request_trace (Id will be there for the label)
        # id_paths =
        request_traces = request_traces + shortlisted_request_trace

        # Only continue if we have at least one trace
        if not request_traces:
            return []

        # Check if we have IDs that were unknown during training and map them to generic variants
        training_mapping = get_label_mapping(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "resources",
                "data",
                "mapping.json",
            )
        )
        mapped_request_traces = []

        def get_mapped_id(id):
            if id in training_mapping:
                return id
            else:
                return 42

        print(request_traces)

        for trace in request_traces:
            mapped_trace = []
            for element in trace:
                mapped_trace.append(
                    (
                        get_mapped_id(element[0]),
                        [get_mapped_id(id) for id in element[1]],
                        [get_mapped_id(id) for id in element[2]],
                    )
                )
            mapped_request_traces.append(mapped_trace)
        print("Mapped request traces:{}".format(mapped_request_traces))
        # Create predictions and combine them to a single prediction
        summed_prediction = None
        # print("Mapped request trace:{}".format(mapped_request_traces))
        # Throw out invalid predictions and build final ranked prediction set
        for trace in mapped_request_traces:
            # Convert trace-predecessor-successor triple to three dimensional list for predictor
            trace_list = [
                [e[0] for e in trace],
                [e[1] for e in trace],
                [e[2] for e in trace],
            ]
            prediction = np.array(self.predict_next_element(trace_list, True))

            if summed_prediction is None:
                summed_prediction = np.zeros(len(prediction))
            summed_prediction = np.add(summed_prediction, prediction)
        # print("Summed prediction:{}".format(summed_prediction))
        mean_prediction = np.true_divide(summed_prediction, len(mapped_request_traces))
        # print("Mean_prediction prediction len:{}".format(len(mean_prediction)))
        # print("Mean_prediction:{}".format(mean_prediction))

        # Throw out invalid predictions and build final ranked prediction set
        final_predictions = []

        # Code to get all the unique types
        # type_set = set()

        for i in range(0, len(mean_prediction)):
            # print(training_mapping.keys())
            if i in training_mapping.keys():
                type = training_mapping[i]["type"]
                # type_set.add(type)
                if mean_prediction[i] > 0.1:
                    final_predictions.append(
                        Prediction(
                            training_mapping[i]["label"],
                            Type(type[0].upper() + type[1:]),
                            # None,
                            mean_prediction[i],
                        )
                    )
        # print("Total number of final predictions:{}".format(len(final_predictions)))
        # print("Unique type list:{}".format(type_set))
        return final_predictions

    def predict_next_element(self, data, get_probabilities):
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
            print(formatted_data)
            prediction = self.model.predict(formatted_data)
            # print("Prediction:{}".format(prediction))
            if get_probabilities:
                print(prediction[0, -1])
                return prediction[0, -1]
            else:
                return np.argmax(prediction[0, -1])
        return None

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
