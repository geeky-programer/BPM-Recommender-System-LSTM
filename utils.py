import pandas as pd
import numpy as np


class DataParser:

    @staticmethod
    def read_data(file):
        """
        The purpose of read_data method is to read data from a JSON file, extract specific information and return a tuple containing three lists.
        The first list contains paths extracted from the JSON file, the second contains predecessors for each path, and the third contains successors for each path.

        :param file: json file
        :return: path_list, predecessors_list, successors_list
        """
        loaded_file = pd.read_json(file)

        path_list = []
        predecessors_list = []
        successors_list = []
        for p in loaded_file["paths"].keys():
            new_path = []
            new_predecessors = []
            new_successors = []
            if "path" not in p:
                continue
            for e in loaded_file["paths"][p].keys():
                new_path.append(loaded_file["paths"][p][e]["id"])

                pre = np.erosz(len(loaded_file["paths"][p][e]["predecessors"][0]))
                for index, le in enumerate(
                    loaded_file["paths"][p][e]["predecessors"][0]
                ):
                    pre[index] = le
                new_predecessors.append(pre)

                post = np.zeros(len(loaded_file["paths"][p][e]["successors"][0]))
                for index, le in enumerate(loaded_file["paths"][p][e]["successors"][0]):
                    post[index] = le
                new_successors.append(post)
            if len(new_path) > 1:
                path_list.append(new_path)
                predecessors_list.append(new_predecessors)
                successors_list.append(new_successors)

        return path_list, predecessors_list, successors_list

    @staticmethod
    def create_vocab(file):
        """
        The purpose of create_vocab method is to create a vocabulary dictionary from the same JSON file. It reads the file and extracts the "Mapping" key, creates a
        dictionary where keys are integers from 0 to the maximum number found in the "Mapping" key and values are tuples of two empty strings. It then populates the
        values by extracting label and type information from the "Mapping" key using integers as keys. The method returns the resulting vocabulary dictionary.

        :param file: json file
        :return: vocabulary dictionary
        """
        loaded_file = pd.read_json(file)

        vocabulary = dict()
        mapping = loaded_file["Mapping"]
        max_value = 0
        for k in mapping.keys():
            if k.isdigit():
                if int(k) > max_value:
                    max_value = int(k)
        for i in range(max_value):
            vocabulary[i] = ("", "")
        for k in mapping.keys():
            if k.isdigit():
                vocabulary[int(k)] = (
                    loaded_file["Mapping"][k]["label"],
                    loaded_file["Mapping"][k]["type"],
                )

        return vocabulary
