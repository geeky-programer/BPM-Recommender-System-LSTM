## Multi layer LSTM training code.
from __future__ import print_function

import numpy as np

from utils import DataParser
from data_loader import DataLoader

from lstm import Lstm

## Method to prepare data, create and train the model
def prepare_train(
    data_set, num_steps, batch_size, hidden_size, hidden_size_neighbour, neighbor_count, vocabulary, train_paths, valid_data, trial=None, ok=None
):

    # Generate training data
    train_data_generator = DataLoader(
        data_set,
        num_steps,
        vocabulary,
        batch_size,
        True,
        neighbor_count,
        unknown_percentage=0.2,
    )
    # Generate validation data
    valid_data_generator = DataLoader(
        data_set,
        num_steps,
        vocabulary,
        batch_size,
        True,
        neighbor_count,
        unknown_percentage=0.2
    )

    # Create and define LSTM model
    model_wrapper = Lstm()
    model_wrapper.create(
        vocabulary,
        hidden_size,
        hidden_size_neighbour,
        True,
        neighbor_count
    )

    model_wrapper.load()
    print(model_wrapper.model.summary())

    # Define epochs and train the model
    num_epochs = 50  # was 50
    model_wrapper.train(
        train_data_generator,
        valid_data_generator,
        len(train_paths),
        len(valid_data),
        num_epochs
    )

    return model_wrapper


# Read the data from mapping.json file
train_data_file = "mapping.json"
(train_paths, train_predecessors, train_successors) = DataParser.read_data(train_data_file)

# Create vocabulary - Assign unique numbers for different nodes/labels

vocabulary = DataParser.create_vocab(train_data_file)
valid_data = train_paths.copy()


# Defining hyperparameters
num_steps = 10
batch_size = 8
hidden_size = 32
hidden_size_neighbour = 32

# Train the model
model = prepare_train(
	[train_paths, train_predecessors, train_successors],
    num_steps,
    batch_size,
	vocabulary,
	train_paths,
	valid_data
)

# Test the prediction
example_training_generator = DataLoader(
    [train_paths, train_predecessors, train_successors],
    num_steps,
    vocabulary,
    batch_size,
    True,
    unknown_percentage=0.2,
)

print("Prediction Test:")
num_predict = 2
for i in range(num_predict):
    data = next(example_training_generator.generate())
    input_data = [data[0][0][2], data[0][1][2], data[0][2][2]]
    trace = np.append(data[0][0][2], np.argmax(data[1][0][-1]))

    prediction = model.predict_trace(input_data)
    predicted_trace = np.append(data[0][0][2][0], prediction)
    print("Test #{}: Original Trace: {}".format(i + 1, trace.__str__()))
    print("Test #{}: Predicted Trace: {}".format(i + 1, predicted_trace.__str__()))
