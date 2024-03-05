import numpy as np
import os 
from keras.callbacks import EarlyStopping
from utilities import load_object
from RNN_utilities import get_XY, create_GRU, create_LSTM, create_RNN, print_predictions

num_props = 1
 
ws_range = range(2, 7)

hidden_range = range(20, 120, 20)

model_list = ["LSTM", "RNN", "GRU"] 

for filename in os.listdir("actual_train"):

    varname = filename.replace("actual_train_", "")

    file_object_train = load_object("actual_train/actual_train_" + varname)
    file_object_val = load_object("actual_val/actual_val_" + varname)
    file_object_test = load_object("actual/actual_" + varname)

    for model_name in model_list:

        for ws_use in ws_range:
            
            x_train_all = []
            y_train_all = []

            for k in file_object_train:

                x_train_part, y_train_part = get_XY(file_object_train[k], ws_use)
                
                for ix in range(len(x_train_part)):
                    x_train_all.append(x_train_part[ix]) 
                    y_train_all.append(y_train_part[ix])

            x_train_all = np.array(x_train_all)
            y_train_all = np.array(y_train_all)
            
            x_test_all = []
            y_test_all = []

            for k in file_object_test:

                x_test_part, y_test_part = get_XY(file_object_test[k], ws_use)
                
                for ix in range(len(x_test_part)):
                    x_test_all.append(x_test_part[ix]) 
                    y_test_all.append(y_test_part[ix])

            x_test_all = np.array(x_test_all)
            y_test_all = np.array(y_test_all)
            
            x_val_all = []
            y_val_all = []

            for k in file_object_val:

                x_val_part, y_val_part = get_XY(file_object_val[k], ws_use)
                
                for ix in range(len(x_val_part)):
                    x_val_all.append(x_val_part[ix]) 
                    y_val_all.append(y_val_part[ix])

            x_val_all = np.array(x_val_all)
            y_val_all = np.array(y_val_all)

            for hidden_use in hidden_range:
 
                if model_name == "RNN":
                    demo_model = create_RNN(hidden_use, ws_use, (ws_use, num_props)) 

                if model_name == "GRU": 
                    demo_model = create_GRU(hidden_use, ws_use, (ws_use, num_props)) 

                if model_name == "LSTM": 
                    demo_model = create_LSTM(hidden_use, ws_use, (ws_use, num_props)) 
            
                if not os.path.isdir("train_net/" + varname + "/models/" + model_name):
                    os.makedirs("train_net/" + varname + "/models/" + model_name)

                if not os.path.isdir("train_net/" + varname + "/predictions/train/" + model_name):
                    os.makedirs("train_net/" + varname + "/predictions/train/" + model_name)
            
                if not os.path.isdir("train_net/" + varname + "/predictions/test/" + model_name):
                    os.makedirs("train_net/" + varname + "/predictions/test/" + model_name)
            
                if not os.path.isdir("train_net/" + varname + "/predictions/val/" + model_name):
                    os.makedirs("train_net/" + varname + "/predictions/val/" + model_name)
            
                demo_model.save("train_net/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + ".h5") 
                callback = [EarlyStopping(monitor = 'loss', mode= 'min', patience = 10)]
                history_model = demo_model.fit(x_train_all, y_train_all, verbose = 0, epochs = 70, batch_size = 600, callbacks = callback)
                
                predict_train_all = demo_model.predict(x_train_all)
                print_predictions(y_train_all, predict_train_all, "train_net/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_train.csv") 
                
                predict_val_all = demo_model.predict(x_val_all)
                print_predictions(y_val_all, predict_val_all, "train_net/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_val.csv")     
        
                predict_test_all = demo_model.predict(x_test_all)
                print_predictions(y_test_all, predict_test_all, "train_net/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv")