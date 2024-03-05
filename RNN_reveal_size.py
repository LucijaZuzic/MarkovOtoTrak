import os  
from utilities import load_object 

num_props = 1
 
ws_range = [2]

hidden_range = range(120, 220, 20)

model_list = ["LSTM", "RNN", "GRU"] 

for filename in os.listdir("actual_train"):
    
    varname = filename.replace("actual_train_", "")

    file_object_train = load_object("actual_train/actual_train_" + varname)
    file_object_val = load_object("actual_val/actual_val_" + varname)
    file_object_test = load_object("actual/actual_" + varname)
    
    file_object_train_lens = [len(file_object_train[k]) for k in file_object_train]
    file_object_val_lens = [len(file_object_val[k]) for k in file_object_val]
    file_object_test_lens = [len(file_object_test[k]) for k in file_object_test]

    print(min(min(min(file_object_train_lens), min(file_object_val_lens)), min(file_object_test_lens)))
    