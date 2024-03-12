import numpy as np
import os 
import math
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utilities import load_object
from pytorch_utilities import get_XY, print_predictions, PyTorchGRUModel, PyTorchLSTMModel, PyTorchRNNModel

num_props = 1

for varname in os.listdir("train_pytorch"): 
 
    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)

    file_object_train = load_object("actual_train/actual_train_" + varname)
    file_object_val = load_object("actual_val/actual_val_" + varname)
    file_object_train_val = load_object("actual_train_val/actual_train_val_" + varname)
    file_object_test = load_object("actual/actual_" + varname)

    for model_name in os.listdir("train_pytorch/" + varname + "/predictions/val/"):
        
        ws_array = []
        hidden_array = []
        val_RMSE = []
 
        for filename in os.listdir("train_pytorch/" + varname + "/predictions/val/" + model_name):

            val_data = pd.read_csv("train_pytorch/" + varname + "/predictions/val/" + model_name + "/" + filename, sep = ";", index_col = False)
             
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
            pytorch_model = PyTorchRNNModel(ws_use, hidden_use, ws_use)

        if model_name == "GRU": 
            pytorch_model = PyTorchGRUModel(ws_use, hidden_use, ws_use)

        if model_name == "LSTM": 
            pytorch_model = PyTorchLSTMModel(ws_use, hidden_use, ws_use)
    
        if not os.path.isdir("final_train_pytorch/" + varname + "/models/" + model_name):
            os.makedirs("final_train_pytorch/" + varname + "/models/" + model_name)

        if not os.path.isdir("final_train_pytorch/" + varname + "/predictions/train/" + model_name):
            os.makedirs("final_train_pytorch/" + varname + "/predictions/train/" + model_name)
    
        if not os.path.isdir("final_train_pytorch/" + varname + "/predictions/test/" + model_name):
            os.makedirs("final_train_pytorch/" + varname + "/predictions/test/" + model_name)
    
        if not os.path.isdir("final_train_pytorch/" + varname + "/predictions/val/" + model_name):
            os.makedirs("final_train_pytorch/" + varname + "/predictions/val/" + model_name)
    
        pytorch_model.eval()

        torch.save(pytorch_model.state_dict(), "final_train_pytorch/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + ".pth")

        train_dataset = TensorDataset(torch.tensor(x_train_val_all).float(),  torch.tensor(y_train_val_all).float())
        train_loader = DataLoader(train_dataset, batch_size=600, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(pytorch_model.parameters())

        num_epochs = 70
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = pytorch_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        pytorch_model.eval()

        with torch.no_grad():

            predict_train_all = pytorch_model(torch.tensor(x_train_all).float())

            predict_val_all = pytorch_model(torch.tensor(x_val_all).float())

            predict_test_all = pytorch_model(torch.tensor(x_test_all).float())
            
            print_predictions(y_train_all, predict_train_all, "final_train_pytorch/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_train.csv") 
            
            print_predictions(y_val_all, predict_val_all, "final_train_pytorch/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_val.csv")     
    
            print_predictions(y_test_all, predict_test_all, "final_train_pytorch/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv")