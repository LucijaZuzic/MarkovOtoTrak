import pandas as pd
import os  
from utilities import load_object , save_object
from sacrebleu.metrics import BLEU
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

import numpy as np

predicted_all = dict()
actual_all = dict()
y_test_all = dict()
ws_all = dict() 
BLEU_all = dict()

for varname in os.listdir("train_attention4"):
    
    print(varname)

    final_train_RMSE = []
    final_test_RMSE = []
    final_val_RMSE = []

    final_train_R2 = []
    final_test_R2 = []
    final_val_R2 = []

    final_train_MAE = []
    final_test_MAE = []
    final_val_MAE = []

    test_ix = []
    
    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)
             
    model_name = "GRU_Att"
    ws_use = 1

    for test_num in range(1, 5):

        #print(test_num)
 
        final_val_data = pd.read_csv("train_attention" + str(test_num) + "/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val.csv", sep = ";", index_col = False)
        final_val_data_predicted = [str(x).split(" ")[0].replace("a", ".") for x in final_val_data["predicted"]]
        
        final_val_data_new = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", sep = ">", index_col = False)
        final_val_data_actual = [float(str(y).split(" ")[0].replace("a", ".")) for y in final_val_data_new["y"]]

        final_train_data = pd.read_csv("train_attention" + str(test_num) + "/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train.csv", sep = ";", index_col = False)
        final_train_data_predicted = [str(x).split(" ")[0].replace("a", ".") for x in final_train_data["predicted"]]
        
        final_train_data_new = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv", sep = ">", index_col = False)
        final_train_data_actual = [float(str(y).split(" ")[0].replace("a", ".")) for y in final_train_data_new["y"]]

        final_test_data = pd.read_csv("train_attention" + str(test_num) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv", sep = ";", index_col = False)
        final_test_data_predicted = [str(x).split(" ")[0].replace("a", ".") for x in final_test_data["predicted"]]
        
        final_test_data_new = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv", sep = ">", index_col = False)
        final_test_data_actual = [float(str(y).split(" ")[0].replace("a", ".")) for y in final_test_data_new["y"]]
    
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
        final_val_R2.append(r2_score(final_val_data_actual, final_val_data_predicted))
        final_val_RMSE.append(math.sqrt(mean_squared_error(final_val_data_actual, final_val_data_predicted)) / (max(all_mine_flat) - min(all_mine_flat)))

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
        final_train_R2.append(r2_score(final_train_data_actual, final_train_data_predicted))
        final_train_RMSE.append(math.sqrt(mean_squared_error(final_train_data_actual, final_train_data_predicted)) / (max(all_mine_flat) - min(all_mine_flat)))

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
        final_test_R2.append(r2_score(final_test_data_actual, final_test_data_predicted))
        final_test_RMSE.append(math.sqrt(mean_squared_error(final_test_data_actual, final_test_data_predicted)) / (max(all_mine_flat) - min(all_mine_flat)))
 
        #print(train_unk, len(final_train_data_predicted), np.round(train_unk / len(final_train_data_predicted) * 100, 4))
        
        #print(val_unk, len(final_val_data_predicted), np.round(val_unk / len(final_val_data_predicted) * 100, 4))
        
        #print(test_unk, len(final_test_data_predicted), np.round(test_unk / len(final_test_data_predicted) * 100, 4))
        
        test_ix.append(test_num)

    #print(final_train_RMSE)
    #print(final_val_RMSE)
    #print(final_test_RMSE)

    #for val in final_val_RMSE:
        #print(np.round(val * 100, 2))

    mini_ix_val = final_val_RMSE.index(min(final_val_RMSE))
    mini_ix_test = final_test_RMSE.index(min(final_test_RMSE))

    print(mini_ix_val, test_ix[mini_ix_val], final_val_RMSE[mini_ix_val], final_test_RMSE[mini_ix_val])
    print(np.round(final_test_RMSE[mini_ix_val] * 100, 2), np.round(final_test_R2[mini_ix_val] * 100, 2), np.round(final_test_MAE[mini_ix_val], 6))
    #print(mini_ix_test, test_ix[mini_ix_test], final_test_RMSE[mini_ix_test])

    predicted_all[varname] = dict()
    actual_all[varname] = dict()
    y_test_all[varname] = dict()
    ws_all[varname] = dict() 
    BLEU_all[varname] = dict()
    
    for model_name in os.listdir("train_attention" + str(test_ix[mini_ix_val]) + "/" + varname + "/predictions/test/"):

        predicted_all[varname][model_name] = dict()
        actual_all[varname][model_name] = dict()
        y_test_all[varname][model_name] = dict() 
        BLEU_all[varname][model_name] = []

        for filename in os.listdir("train_attention" + str(test_ix[mini_ix_val]) + "/" + varname + "/predictions/test/" + model_name):
 
            final_test_data = pd.read_csv("train_attention" + str(test_ix[mini_ix_val]) + "/" + varname + "/predictions/test/" + model_name + "/" + filename, sep = ";", index_col = False)
  
            file_object_test = load_object("actual/actual_" + varname)

            ws_use = int(filename.replace(".csv", "").split("_")[-2])
            ws_all[varname][model_name] = ws_use
 
            len_total = 0

            for k in file_object_test:

                x_test_part, y_test_part = get_XY(file_object_test[k], ws_use)
                
                y_test_all[varname][model_name][k] = []
                
                for ix1 in range(len(y_test_part)): 
                    for ix2 in range(len(y_test_part[ix1])): 
                        y_test_all[varname][model_name][k].append(y_test_part[ix1][ix2])

                final_test_data_predicted = [str(x).split(" ")[0] for x in final_test_data["predicted"]]
                final_test_data_actual = [str(x).split(" ")[0] for x in final_test_data["actual"]]
   
                predicted_all[varname][model_name][k] = list(final_test_data_predicted[len_total:len_total + len(y_test_all[varname][model_name][k])])
                actual_all[varname][model_name][k] = list(final_test_data_actual[len_total:len_total + len(y_test_all[varname][model_name][k])])

                bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
                bleu = BLEU(**bleu_params)
                pred_str = ""
                actual_str = "" 
                for val_ix in range(len(predicted_all[varname][model_name][k])):
                    pred_str += str(predicted_all[varname][model_name][k]) + " "
                    actual_str += str(actual_all[varname][model_name][k]) + " "
                pred_str = pred_str[:-1]
                actual_str = actual_str[:-1] 
                blsc = bleu.sentence_score(hypothesis=pred_str, references=[actual_str]).score
                BLEU_all[varname][model_name].append(blsc)
                print(varname, model_name, k, BLEU_all[varname][model_name][-1])
                save_object("attention_result/BLEU_all", BLEU_all) 
        print(varname, model_name, np.mean(BLEU_all[varname][model_name]))

for varname in BLEU_all:
    
    for model_name in BLEU_all[varname]:

        print(varname, model_name, np.mean(BLEU_all[varname][model_name]))