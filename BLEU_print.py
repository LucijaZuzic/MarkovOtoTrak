from utilities import load_object
import numpy as np
BLEU_all = load_object("attention_result/BLEU_all")
for varname in BLEU_all:
    
    for model_name in BLEU_all[varname]:

        print(varname, model_name, np.mean(BLEU_all[varname][model_name]))