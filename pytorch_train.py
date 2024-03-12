import numpy as np
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utilities import load_object
from pytorch_utilities import get_XY, print_predictions, PyTorchGRUModel, PyTorchLSTMModel, PyTorchRNNModel

num_props = 1
 
ws_range = range(2, 7)

hidden_range = range(20, 120, 20)

model_list = ["GRU"] 

for filename in os.listdir("actual_train"):

    varname = filename.replace("actual_train_", "")

    if "direction" in varname:
        continue

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
                    pytorch_model = PyTorchRNNModel(ws_use, hidden_use, ws_use)

                if model_name == "GRU": 
                    pytorch_model = PyTorchGRUModel(ws_use, hidden_use, ws_use)

                if model_name == "LSTM": 
                    pytorch_model = PyTorchLSTMModel(ws_use, hidden_use, ws_use)

                if not os.path.isdir("train_pytorch/" + varname + "/models/" + model_name):
                    os.makedirs("train_pytorch/" + varname + "/models/" + model_name)

                if not os.path.isdir("train_pytorch/" + varname + "/predictions/train/" + model_name):
                    os.makedirs("train_pytorch/" + varname + "/predictions/train/" + model_name)
            
                if not os.path.isdir("train_pytorch/" + varname + "/predictions/test/" + model_name):
                    os.makedirs("train_pytorch/" + varname + "/predictions/test/" + model_name)
            
                if not os.path.isdir("train_pytorch/" + varname + "/predictions/val/" + model_name):
                    os.makedirs("train_pytorch/" + varname + "/predictions/val/" + model_name)
            
                pytorch_model.eval()

                torch.save(pytorch_model.state_dict(), "train_pytorch/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + ".pth")

                train_dataset = TensorDataset(torch.tensor(x_train_all).float(),  torch.tensor(y_train_all).float())
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
                    
                    print_predictions(y_train_all, predict_train_all, "train_pytorch/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_train.csv") 
                    
                    print_predictions(y_val_all, predict_val_all, "train_pytorch/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_val.csv")     
            
                    print_predictions(y_test_all, predict_test_all, "train_pytorch/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv")