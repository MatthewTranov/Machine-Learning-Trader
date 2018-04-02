import ast
import numpy as np
import pandas as pd


from keras.layers import LSTM, Input, Dense
from keras.models import Model

WINDOW_SIZE = 5

def get_data(file_path):
    '''turn file of lists into list of lists'''
    stock_list = []
    open_file = open(file_path, 'r')

    for line in open_file:
        try:
            price_every_five_minutes_list = ast.literal_eval(line)
        except:
            print(line)
        stock_list.append(price_every_five_minutes_list)
    open_file.close()
    return stock_list

def price_data_to_percent_data(data):
    '''Convert from stock prices to percent change from last price'''

    list_of_percent_change_lists = []
    # get the list of prices of each stock
    for stock_prices in data:
        percent_change_list = []
        # goes through the index of every stock except the first one
        for i in range(1, len(stock_prices)):
            # caculate percent the stock went up or down after last 5 minutes
            if stock_prices[i-1] == 0:
                percent_change = 0
            else:
                percent_change = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
            # round to 100th of a percent
            percent_change = round(percent_change, 4)
            # add to list
            percent_change_list.append(percent_change)
        list_of_percent_change_lists.append(percent_change_list)
    return list_of_percent_change_lists

def write_data(data, file_path):
    file = open(file_path, 'a')
    for element in data:
        file.write(str(element)+'\n')
    file.close()

def preprocess_data(price_data):
    'turn percent change data into windows and following prices'
    given_data = []
    future_data = []
    for stock in price_data:
        # go through each window of prices not counting the last one which
        # can't be used for prediction
        price_tuples_list = []
        next_change_list = []
        for i in range(len(stock) - WINDOW_SIZE - 1):
            # add the window to the list
            price_tuples_list.append(np.array(stock[i:i + WINDOW_SIZE],ndmin =3))
            # add the next price to the list
            next_change_list.append(np.array(stock[i + WINDOW_SIZE],ndmin =3))
        given_data.append(price_tuples_list)
        future_data.append(next_change_list)
    return given_data, future_data
    # output the data to use and the data to predict

def fix_data(percent_change_data):
    better_list = []
    for stock_percent_change_data in percent_change_data:
        should_delete = False
        countains_non_zero = False
        i = 0
        while not should_delete and i < len(stock_percent_change_data):
            if stock_percent_change_data[i] != 0:
                countains_non_zero = True
            if abs(stock_percent_change_data[i]) >= 0.1:
                should_delete = True
            i += 1
        if not should_delete and countains_non_zero:
            better_list.append(stock_percent_change_data)

    return better_list

def evaulate_fitness(data, model):
    pass
    # go through each stock

    # go through each window of data the model is evaluating

    # if the model is not currently bought in

    # if the model buy neuron is higher than the sell or hold neuron then get
    # the price the model bought at

    # if the model is bought in and the sell value is higher than the hold or
    # sell neuron then get the percent change of the stocks value between buy
    # and sell

    # return the total profit gain as the output


def data_creation():
    # get stock prices

    stock_lists = get_data('stock_data.txt')

    # get percent change
    list_of_percent_change_list = preprocess_data(stock_lists)

    # write percent change
    write_data(list_of_percent_change_list ,'percent_change_data.txt')


def neural_net(given_data, future_data):
    x_train = given_data[:2000]
    x_eval = given_data[2000:3000]
    y_train = future_data[:2000]
    y_eval = future_data[2000:3000]

    # tell keras to expect an unknown number of window size-dimensional vectors
    inputs = Input(batch_shape=(1, 1, WINDOW_SIZE))
    # creates LSTM to act as second layer of neural net
    # the output is one unit
    # the previous layer (inputs) is taken in as input
    encoder = LSTM(5, return_sequences=True, stateful=True)(inputs)
    # creates a final layer of the neural net to act as output
    predictions = Dense(1, activation='sigmoid')(encoder)
    # use first and last layer of neural net as input and output
    model = Model(inputs=inputs, outputs=predictions)
    # configures the model for training
    model.compile(optimizer='adam', loss='mean_squared_error')

    print('Train...')
    # train on the data 15 times
    for epoch in range(15):
        train_loss_list = []
        for i in range(len(x_train)):
            if i % (len(x_train)//10) == 0:
                print(i // (len(x_train)//10), '/10')
            for j in range(len(x_train[i])):
                tr_loss = model.train_on_batch(x_train[i][j], y_train[i][j])
                train_loss_list.append(tr_loss)
            model.reset_states()
        prediction = model.predict(np.array([-0.0028, -0.0029, -0.0028, -0.0007, 0.0014],ndmin =3))
        print(prediction)
        print('loss on training data = {}'.format(np.mean(train_loss_list)))
        print('___________________________________')

        test_loss_list = []
        for i in range(len(x_eval)):
            if i % (len(x_eval) // 10) == 0:
                print(i // (len(x_eval)//10), '/10')
            for j in range(len(x_eval[i])):
                te_loss = model.test_on_batch(x_eval[i][j], y_eval[i][j])
                test_loss_list.append(te_loss)
            model.reset_states()

        print('loss on test data = {}'.format(np.mean(test_loss_list)))
        print('___________________________________')

percent_lists = get_data('fixed_percent_change_data.txt')
#fixed_list = fix_data(percent_lists)
#write_data(fixed_list, 'fixed_percent_change_data.txt')
X_data, Y_data = preprocess_data(percent_lists)
neural_net(X_data, Y_data)

