from utilities import load_object
import os
import numpy as np
from datetime import timedelta, datetime

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

ws_range = [1]

for ws_use in ws_range:
    
    yml_part = "task_dataset:"
    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")

        file_object_train = load_object("actual_train/actual_train_" + varname) 
        file_object_val = load_object("actual_val/actual_val_" + varname)
        file_object_test = load_object("actual/actual_" + varname)

        dictio = {"task_name": "long_term_forecast", "frequency": "1S", "dataset_name": varname, "dataset": varname, "data": "custom", "root_path": "dataset_new/" + varname, "data_path": "newdata_ALL.csv", "seq_len": ws_use, "label_len": 0, "pred_len": ws_use, "features": "M", "embed": "timeF", "enc_in": 767, "dec_in": 767, "c_out": 767}

        yml_part += "\n " + str(varname) + ":"
        for v in dictio:
            yml_part += "\n  " + v + ": " + str(dictio[v])
        yml_part += "\n"
        continue
        together_csv = "date," + varname + "\n"
    
        str_train = "" 
        str_val = ""
        str_test = ""
        datetime_use = datetime(day = 1, month = 1, year = 1970)
        
        for k in file_object_train:

            x_train_part, y_train_part = get_XY(file_object_train[k], ws_use)
            
            for ix1 in range(len(x_train_part)):
                for ix2 in range(len(x_train_part[ix1])):
                    
                    str_train += datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S") + "," + str(x_train_part[ix1][ix2]).replace(",", ".") + ","
                    datetime_use += timedelta(seconds = 1)

                str_train = str_train[:-1]

                str_train += "\n" 

        for k in file_object_val:

            x_val_part, y_val_part = get_XY(file_object_val[k], ws_use)
            
            for ix1 in range(len(x_val_part)):
                for ix2 in range(len(x_val_part[ix1])):

                    str_val += datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S") + "," + str(x_val_part[ix1][ix2]).replace(",", ".") + ","
                    datetime_use += timedelta(seconds = 1)

                str_val = str_val[:-1]

                str_val += "\n"

        for k in file_object_test:

            x_test_part, y_test_part = get_XY(file_object_test[k], ws_use)
            
            for ix1 in range(len(x_test_part)):
                for ix2 in range(len(x_test_part[ix1])):

                    str_test += datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S") + "," + str(x_test_part[ix1][ix2]).replace(",", ".") + ","
                    datetime_use += timedelta(seconds = 1)

                str_test = str_test[:-1]

                str_test += "\n" 

        if not os.path.isdir("csv_data/dataset/" + varname):
            os.makedirs("csv_data/dataset/" + varname)

        file_train_write = open("csv_data/dataset/" + varname + "/newdata_TRAIN.csv", "w")
        file_train_write.write(together_csv + str_train)
        file_train_write.close()
        file_train_val_write = open("csv_data/dataset/" + varname + "/newdata_TRAIN_VAL.csv", "w") 
        file_train_val_write.write(together_csv + str_train + str_val)
        file_train_val_write.close()
        file_val_write = open("csv_data/dataset/" + varname + "/newdata_VAL.csv", "w")
        file_val_write.write(together_csv + str_val)
        file_val_write.close()
        file_test_write = open("csv_data/dataset/" + varname + "/newdata_TEST.csv", "w")
        file_test_write.write(together_csv + str_test)
        file_test_write.close()
        file_all_write = open("csv_data/dataset/" + varname + "/newdata_ALL.csv", "w") 
        file_all_write.write(together_csv + str_train + str_val + str_test)
        file_all_write.close()
    
    if not os.path.isdir("csv_data/data_provider"):
        os.makedirs("csv_data/data_provider")

    file_yml_pre_write = open("csv_data/data_provider/multi_task_pretrain.yaml", "w")
    file_yml_pre_write.write(yml_part.replace("long_term_forecast", "pretrain_long_term_forecast"))
    file_yml_pre_write.close()
    file_yml_write = open("csv_data/data_provider/zeroshot_task.yaml", "w")
    file_yml_write.write(yml_part)
    file_yml_write.close()