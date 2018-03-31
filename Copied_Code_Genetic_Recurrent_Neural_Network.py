import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

# create a seed to generate random numbers
np.random.seed(1120)

# load the data
data = pd.read_csv('train.csv')
data = np.reshape(np.array(data['wp1']), (len(data['wp1']), 1))

# Use first 10000 points as training/validation and rest of the 1500 points as
# test set.
train_data = data[0:10000]
test_data = data[10000:]


def prepare_dataset(data, window_size):
    ''''''
    # create empty two empty arrays
    # X has no length and width of the data being looked at by the neural
    # net (window size)
    given_data = np.empty((0, window_size))
    # Y has no width no height
    future_data = np.empty((0))

    # look through the entire data a window at a time
    for i in range(len(data) - window_size - 1):
        # add the data being looked at to the bottom of the current data array
        given_data = np.vstack([given_data, data[i:(i + window_size), 0]])
        # add the data not being looked end of the future array
        future_data = np.append(future_data, data[(i + window_size), 0])
    # turn given data into 3d array one element thick
    given_data = np.reshape(given_data, (len(given_data), window_size, 1))
    # turn future data into 2d array one element wide
    future_data = np.reshape(future_data, (len(future_data), 1))
    return given_data, future_data


def train_evaluate(ga_individual_solution):
    # Decode GA solution to integer for window_size and num_units outside
    # window
    window_size_bits = BitArray(ga_individual_solution[0:6])
    num_units_bits = BitArray(ga_individual_solution[6:])
    # convert window size and num units from binary to decimal
    window_size = window_size_bits.uint
    num_units = num_units_bits.uint
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)

    # if window_size or num_unit is zero then all the data is being read so
    # the net has 100% accuracy
    if window_size == 0 or num_units == 0:
        fitness = 100
    else:
        # Segment the train_data based on new window_size; split into train
        # and validation (80/20)
        given_data, future_data = prepare_dataset(train_data, window_size)
        X_train, X_eval, y_train, y_eval = split(given_data, future_data,
                                                 test_size=0.20,
                                                 random_state=1120)

        # Train LSTM model and predict on validation set

        # tell keras what shape of tensors to expect as input
        # this tensor acts as the first layer of the neural net
        inputs = Input(shape=(window_size, 1))

        # creates LSTM to act as second layer of neural net
        # num units is what to output
        # the previous layer (inputs) is taken in as input
        encoder = LSTM(num_units, stateful=False, input_shape=(window_size,
                                                         1))(inputs)
        # creates a final layer of the neural net to act as output
        predictions = Dense(1, activation='linear')(encoder)
        # use first and last layer of neural net as input and output
        model = Model(inputs=inputs, outputs=predictions)
        # configures the model for training
        model.compile(optimizer='adam', loss='mean_squared_error')
        # train the net on the training model
        model.fit(X_train, y_train, epochs=5, batch_size=10, shuffle=True)
        # test the model on unknown data
        y_pred = model.predict(X_eval)

        # Calculate the RMSE score as fitness score for GA
        fitness = np.sqrt(mean_squared_error(y_eval, y_pred))
        print('Validation RMSE: ', fitness, '\n')

    return fitness


population_size = 4
num_generations = 4
gene_length = 10

# As we are trying to minimize the RMSE score, that's why using -1.0.
# In case, when you want to maximize accuracy for instance, use 1.0

# creates a class like object that inherits from fitness, with -1 weight
# saying the the best fitness is the minimum
creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
# creates a class like object that inherits from list
creator.create('Individual', list, fitness=creator.FitnessMax)

# object that contains evolutionary methods
toolbox = base.Toolbox()
# registers a method called binary in the toolbox that randomly creates a 1
# or 0 (0.5 is the chance of each passed into bernoulli.rvs)
toolbox.register('binary', bernoulli.rvs, 0.5)
# registers a method called individual that calls the binary method multiple
# times to create a random gene
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.binary, n=gene_length)
# registers a method called population that fills a list with randomly
# generated individuals
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# takes in indices of list and modifies the individuals
toolbox.register('mate', tools.cxOrdered)
# registers a method called mutate that randomly shuffles the indexes of an
# individual (indpb is the chance each index is swapped)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6)
# randomly select some number of individuals based on their fitness
toolbox.register('select', tools.selRoulette)
# calculate the fitness of an individual
toolbox.register('evaluate', train_evaluate)
# create a population of individuals with randomly generated genes
population = toolbox.population(n=population_size)
# easimple iss the method to run the evolutionary algorithm it returns the
# final population
r = algorithms.eaSimple(population,  # population to evolve
                        toolbox,  # toolbox that contain evolutionary algorithms
                        cxpb=0.4,  # mating probability
                        mutpb=0.1,  # mutation probability
                        ngen=num_generations,  # num generations
                        verbose=False)  # log statistics or not

# Print highest or lowest fitness individuals (k = 1 means the 1st only)
best_individuals = tools.selBest(population, k=1)

best_window_size = None
best_num_units = None
for bi in best_individuals:
    # Decode GA solution to integer for window_size and num_units outside
    # window
    window_size_bits = BitArray(bi[0:6])
    num_units_bits = BitArray(bi[6:])
    # convert window size and num units from binary to decimal
    best_window_size = window_size_bits.uint
    best_num_units = num_units_bits.uint
    print('\nWindow Size: ', best_window_size, ', Num of Units: ',
          best_num_units)

# Train the model using best configuration on complete training set
# and make predictions on the test set
X_train, y_train = prepare_dataset(train_data, best_window_size)
X_test, y_test = prepare_dataset(test_data, best_window_size)

inputs = Input(shape=(best_window_size, 1))
x = LSTM(best_num_units, input_shape=(best_window_size, 1))(inputs)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=10, shuffle=True)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: ', rmse)
