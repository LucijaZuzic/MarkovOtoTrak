import pandas as pd
import os
from utilities import load_object
import numpy as np
from sklearn.metrics import mean_absolute_error

for varname in os.listdir("train_attention"):
    
    print(varname)

    final_train_MAE = []
    final_test_MAE = []
    final_val_MAE = []
    
    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)
             
    model_name = "GRU_Att"
    ws_use = 1

    final_val_data = pd.read_csv("train_attention/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val.csv", sep = ";", index_col = False)
    final_val_data_predicted = [str(x).split(" ")[0].replace("a", ".") for x in final_val_data["predicted"]]
    final_val_data_actual = [float(str(x).split(" ")[0].replace("a", ".")) for x in final_val_data["actual"]]

    final_train_data = pd.read_csv("train_attention/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train.csv", sep = ";", index_col = False)
    final_train_data_predicted = [str(x).split(" ")[0].replace("a", ".") for x in final_train_data["predicted"]]
    final_train_data_actual = [float(str(x).split(" ")[0].replace("a", ".")) for x in final_train_data["actual"]]

    final_test_data = pd.read_csv("train_attention/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv", sep = ";", index_col = False)
    final_test_data_predicted = [str(x).split(" ")[0].replace("a", ".") for x in final_test_data["predicted"]]
    final_test_data_actual = [float(str(x).split(" ")[0].replace("a", ".")) for x in final_test_data["actual"]]
 
    val_unk = 0
    for i in range(len(final_val_data_predicted)):
        if str(final_val_data_predicted[i]) == '<unk>':
            val_unk += 1
            if i > 0:
                final_val_data_predicted[i] = final_val_data_predicted[i - 1]
            else:
                final_val_data_predicted[i] = 0
        else:
            final_val_data_predicted[i] = float(final_val_data_predicted[i])
 
    final_val_MAE.append(mean_absolute_error(final_val_data_actual, final_val_data_predicted))

    train_unk = 0
    for i in range(len(final_train_data_predicted)):
        if str(final_train_data_predicted[i]) == '<unk>':
            train_unk += 1
            if i > 0:
                final_train_data_predicted[i] = final_train_data_predicted[i - 1]
            else:
                final_train_data_predicted[i] = 0
        else:
            final_train_data_predicted[i] = float(final_train_data_predicted[i])
 
    final_train_MAE.append(mean_absolute_error(final_train_data_actual, final_train_data_predicted))
    
    test_unk = 0
    for i in range(len(final_test_data_predicted)):
        if str(final_test_data_predicted[i]) == '<unk>':
            test_unk += 1
            if i > 0:
                final_test_data_predicted[i] = final_test_data_predicted[i - 1]
            else:
                final_test_data_predicted[i] = 0
        else:
            final_test_data_predicted[i] = float(final_test_data_predicted[i])
 
    final_test_MAE.append(mean_absolute_error(final_test_data_actual, final_test_data_predicted))
        
    print(final_train_MAE)
    print(final_val_MAE)
    print(final_test_MAE)

    for val in final_test_MAE:
        print(np.round(val, 6))

    print(train_unk, len(final_train_data_predicted), np.round(train_unk / len(final_train_data_predicted) * 100, 4))
    
    print(val_unk, len(final_val_data_predicted), np.round(val_unk / len(final_val_data_predicted) * 100, 4))
    
    print(test_unk, len(final_test_data_predicted), np.round(test_unk / len(final_test_data_predicted) * 100, 4))