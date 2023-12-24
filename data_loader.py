from random import shuffle, random

class DataLoader(object):
    """
    The class generates batches of training examples for a machine learning
    """
    def __init__(
        self,
        data,
        num_steps,
        vocabulary,
        batch_size=1,
        shuffle_data=False,
        neighbor_count=10,
        unknown_percentage=0,
    ):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary: Dict = vocabulary
        self.shuffle_data = shuffle_data
        self.neighbor_count = neighbor_count
        if self.shuffle_data:
            c = list(zip(self.data[0], self.data[1], self.data[2]))
            shuffle(c)

            self.data[0], self.data[1], self.data[2] = zip(*c)

        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.path_index = 0
        self.unknown_percentage = unknown_percentage
