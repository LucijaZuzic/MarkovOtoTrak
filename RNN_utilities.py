
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
import numpy as np

def print_predictions(actual, predicted, name_file):
    
    strpr = "actual;predicted\n"
    for ix1 in range(len(actual)):
        for ix2 in range(len(actual[ix1])):
            strpr += str(actual[ix1][ix2]) + ";" + str(predicted[ix1][ix2]) + "\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr.replace("[", "").replace("]", ""))
    file_processed.close()

def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def create_RNN(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential() 
    model.add(SimpleRNN(hidden_units, input_shape = input_shape, activation = "linear"))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def create_GRU(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential() 
    model.add(GRU(hidden_units, input_shape = input_shape, activation = "linear"))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def create_LSTM(hidden_units, dense_units, input_shape, act_layer = "linear", act_layer2 = "linear"):
    model = Sequential() 
    model.add(LSTM(hidden_units, input_shape = input_shape, activation = "linear", recurrent_activation = act_layer2))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model