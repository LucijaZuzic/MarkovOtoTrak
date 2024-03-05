import numpy as np
import os 
import math
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.callbacks import EarlyStopping
from utilities import load_object
from RNN_utilities import get_XY, create_GRU, create_LSTM, create_RNN, print_predictions

num_props = 1

for varname in os.listdir("train_net"): 
 
    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)

    file_object_train = load_object("actual_train/actual_train_" + varname)
    file_object_val = load_object("actual_val/actual_val_" + varname)
    file_object_train_val = load_object("actual_train_val/actual_train_val_" + varname)
    file_object_test = load_object("actual/actual_" + varname)

    for model_name in os.listdir("train_net/" + varname + "/predictions/val/"):
        
        ws_array = []
        hidden_array = []
        val_RMSE = []
 
        for filename in os.listdir("train_net/" + varname + "/predictions/val/" + model_name):

            val_data = pd.read_csv("train_net/" + varname + "/predictions/val/" + model_name + "/" + filename, sep = ";", index_col = False)
             
            is_a_nan = False
            for val in val_data["predicted"]:
                if str(val) == 'nan':
                    is_a_nan = True
                    break

            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
 
            if is_a_nan:
                val_RMSE.append(1000000)
            else: 
                val_RMSE.append(math.sqrt(mean_squared_error(val_data["actual"], val_data["predicted"])) / (max(all_mine_flat) - min(all_mine_flat)))
 
        hidden_use = hidden_array[val_RMSE.index(min(val_RMSE))]
        ws_use = ws_array[val_RMSE.index(min(val_RMSE))]
        
        print(ws_use, hidden_use, min(val_RMSE))
            
        x_train_all = []
        y_train_all = []

        for k in file_object_train:

            x_train_part, y_train_part = get_XY(file_object_train[k], ws_use)
            
            for ix in range(len(x_train_part)):
                x_train_all.append(x_train_part[ix]) 
                y_train_all.append(y_train_part[ix])

        x_train_all = np.array(x_train_all)
        y_train_all = np.array(y_train_all)

        print(np.shape(x_train_all), np.shape(y_train_all))
        
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
        
        x_train_val_all = []
        y_train_val_all = []

        for k in file_object_train_val:

            x_train_val_part, y_train_val_part = get_XY(file_object_train_val[k], ws_use)
            
            for ix in range(len(x_train_val_part)):
                x_train_val_all.append(x_train_val_part[ix]) 
                y_train_val_all.append(y_train_val_part[ix])

        x_train_val_all = np.array(x_train_val_all)
        y_train_val_all = np.array(y_train_val_all)

        if model_name == "RNN":
            demo_model = create_RNN(hidden_use, 1, (ws_use, num_props)) 

        if model_name == "GRU": 
            demo_model = create_GRU(hidden_use, 1, (ws_use, num_props)) 

        if model_name == "LSTM": 
            demo_model = create_LSTM(hidden_use, 1, (ws_use, num_props)) 
    
        if not os.path.isdir("final_train_net/" + varname + "/models/" + model_name):
            os.makedirs("final_train_net/" + varname + "/models/" + model_name)

        if not os.path.isdir("final_train_net/" + varname + "/predictions/train/" + model_name):
            os.makedirs("final_train_net/" + varname + "/predictions/train/" + model_name)
    
        if not os.path.isdir("final_train_net/" + varname + "/predictions/test/" + model_name):
            os.makedirs("final_train_net/" + varname + "/predictions/test/" + model_name)
    
        if not os.path.isdir("final_train_net/" + varname + "/predictions/val/" + model_name):
            os.makedirs("final_train_net/" + varname + "/predictions/val/" + model_name)
    
        demo_model.save("final_train_net/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + ".h5") 
        callback = [EarlyStopping(monitor = 'loss', mode= 'min', patience = 10)]
        history_model = demo_model.fit(x_train_val_all, y_train_val_all, verbose = 0, epochs = 70, batch_size = 600, callbacks = callback)
        
        predict_train_all = demo_model.predict(x_train_all)
        print_predictions(y_train_all, predict_train_all, "final_train_net/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_train.csv") 
        
        predict_val_all = demo_model.predict(x_val_all)
        print_predictions(y_val_all, predict_val_all, "final_train_net/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_val.csv")     

        predict_test_all = demo_model.predict(x_test_all)
        print_predictions(y_test_all, predict_test_all, "final_train_net/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv")