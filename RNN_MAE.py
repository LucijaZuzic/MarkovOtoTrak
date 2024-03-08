import pandas as pd
import os  
import math
from sklearn.metrics import mean_squared_error
from utilities import load_object
import numpy as np
from sklearn.metrics import mean_absolute_error

for varname in os.listdir("train_net"):
    
    print(varname)

    final_train_MAE = []
    final_test_MAE = []
    final_val_MAE = []
    
    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)
            
    for model_name in os.listdir("train_net/" + varname + "/predictions/val/"):

        print(model_name)
        
        ws_array = []
        hidden_array = []
        train_RMSE = []
        test_RMSE = []
        val_RMSE = []
 
        for filename in os.listdir("train_net/" + varname + "/predictions/val/" + model_name):

            val_data = pd.read_csv("train_net/" + varname + "/predictions/val/" + model_name + "/" + filename, sep = ";", index_col = False)
             
            train_data = pd.read_csv("train_net/" + varname + "/predictions/train/" + model_name + "/" + filename.replace("val", "train"), sep = ";", index_col = False)
            
            test_data = pd.read_csv("train_net/" + varname + "/predictions/test/" + model_name + "/" + filename.replace("val", "test"), sep = ";", index_col = False)
 
            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
 
            is_a_nan = False
            for val in val_data["predicted"]:
                if str(val) == 'nan':
                    is_a_nan = True
                    break

            if is_a_nan:
                val_RMSE.append(1000000)
            else: 
                val_RMSE.append(math.sqrt(mean_squared_error(val_data["actual"], val_data["predicted"])) / (max(all_mine_flat) - min(all_mine_flat)))
        
            is_a_nan = False
            for val in train_data["predicted"]:
                if str(val) == 'nan':
                    is_a_nan = True
                    break
                
            if is_a_nan:
                train_RMSE.append(1000000)
            else: 
                train_RMSE.append(math.sqrt(mean_squared_error(train_data["actual"], train_data["predicted"])) / (max(all_mine_flat) - min(all_mine_flat)))
            
            is_a_nan = False
            for val in test_data["predicted"]:
                if str(val) == 'nan':
                    is_a_nan = True
                    break
                
            if is_a_nan:
                test_RMSE.append(1000000)
            else: 
                test_RMSE.append(math.sqrt(mean_squared_error(test_data["actual"], test_data["predicted"])) / (max(all_mine_flat) - min(all_mine_flat)))
        
        hidden_use = hidden_array[val_RMSE.index(min(val_RMSE))]
        ws_use = ws_array[val_RMSE.index(min(val_RMSE))]
        train_RMSE_use = train_RMSE[val_RMSE.index(min(val_RMSE))]
        test_RMSE_use = test_RMSE[val_RMSE.index(min(val_RMSE))]

        print(ws_use, hidden_use, min(val_RMSE), train_RMSE_use, test_RMSE_use)
 
        final_val_data = pd.read_csv("final_train_net/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_val.csv", sep = ";", index_col = False)
            
        final_train_data = pd.read_csv("final_train_net/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_train.csv", sep = ";", index_col = False)
        
        final_test_data = pd.read_csv("final_train_net/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv", sep = ";", index_col = False)
 
        is_a_nan = False
        for val in final_val_data["predicted"]:
            if str(val) == 'nan':
                is_a_nan = True
                break

        if is_a_nan:
            final_val_MAE.append(1000000)
        else: 
            final_val_MAE.append(mean_absolute_error(final_val_data["actual"], final_val_data["predicted"]))
    
        is_a_nan = False
        for val in final_train_data["predicted"]:
            if str(val) == 'nan':
                is_a_nan = True
                break
            
        if is_a_nan:
            final_train_MAE.append(1000000)
        else: 
            final_train_MAE.append(mean_absolute_error(final_train_data["actual"], final_train_data["predicted"]))
        
        is_a_nan = False
        for val in final_test_data["predicted"]:
            if str(val) == 'nan':
                is_a_nan = True
                break
            
        if is_a_nan:
            final_test_MAE.append(1000000)
        else: 
            final_test_MAE.append(mean_absolute_error(final_test_data["actual"], final_test_data["predicted"]))
        
    print(final_train_MAE)
    print(final_val_MAE)
    print(final_test_MAE)

    for val in final_test_MAE:
        print(np.round(val * 10000, 2))