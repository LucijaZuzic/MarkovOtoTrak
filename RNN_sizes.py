import os
from utilities import load_object

for filename in os.listdir("actual_train"):

    varname = filename.replace("actual_train_", "")

    file_object_train = load_object("actual_train/actual_train_" + varname)
    file_object_val = load_object("actual_val/actual_val_" + varname)
    file_object_test = load_object("actual/actual_" + varname)
 
    list_keys_train = [int(lk.split("/")[0].split("_")[1]) for lk in list(file_object_train.keys())]
    list_keys_val = [int(lk.split("/")[0].split("_")[1]) for lk in list(file_object_val.keys())]
    list_keys_test = [int(lk.split("/")[0].split("_")[1]) for lk in list(file_object_test.keys())]

    count_keys_train = {k: list_keys_train.count(k) for k in range(1, 20)}
    count_keys_val = {k: list_keys_val.count(k) for k in range(1, 20)}
    count_keys_test = {k: list_keys_test.count(k) for k in range(1, 20)}

    for k in range(1, 20):
        if count_keys_train[k] + count_keys_val[k] + count_keys_test[k] > 0:
            print(k, "&", count_keys_train[k], "&", count_keys_val[k], "&", count_keys_train[k] + count_keys_val[k], "&", count_keys_test[k], "&", count_keys_train[k] + count_keys_val[k] + count_keys_test[k], "\\\\ \\hline",)
    print("All", "&", sum(list(count_keys_train.values())), "&", sum(list(count_keys_val.values())), "&", sum(list(count_keys_train.values())) + sum(list(count_keys_val.values())), "&", sum(list(count_keys_test.values())), "&", sum(list(count_keys_train.values())) + sum(list(count_keys_val.values())) + sum(list(count_keys_test.values())), "\\\\ \\hline")
    break

