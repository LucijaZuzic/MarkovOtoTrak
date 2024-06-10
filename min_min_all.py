from utilities import load_object 
import os 
import numpy as np
translate_name = {"direction": "Heading", "latitude_no_abs": "$y$ offset", "longitude_no_abs": "$x$ offset", "speed" : "Speed", "time" : "Time"}

def str_convert(val):
    if val == False:
        return "0"
    if val == True:
        return "1"
    new_val = val
    power_to = 0
    while abs(new_val) < 1 and new_val != 0.0:
        new_val *= 10
        power_to += 1 
    rounded = str(np.round(new_val, 2))
    if rounded[-2:] == '.0':
        rounded = rounded[:-2]
    if power_to != 0:  
        rounded += " \\times 10^{-" + str(power_to) + "}"
    return rounded

def get_var(name_of, p1 = False, p2 = False, p3 = False):   
    if p1: 
        probability_of = load_object("probability/probability_of_" + name_of) 
        mini_prob = dict()
        for k in probability_of:
            if k == "undefined":
                continue
            mini_prob[k] = probability_of[k]
        lk = list(mini_prob.keys())
        lv = list(mini_prob.values())
        print(translate_name[name_of] + " & $" + str(lk[np.argmin(lv)]) + "$ & $" + str_convert(min(lv) * (10 ** 6)) + "$ \\\\ \\hline")
    if p2:
        probability_of_in_next_step = load_object("probability/probability_of_" + name_of + "_in_next_step")  
        mini_prob2 = dict()
        for k1 in probability_of_in_next_step:
            if k1 == "undefined":
                continue
            for k2 in probability_of_in_next_step[k1]:
                if k2 == "undefined":
                    continue
                mini_prob2[str_convert(k1) + "$ & $" + str_convert(k2)] = probability_of_in_next_step[k1][k2]
        lk2 = list(mini_prob2.keys())
        lv2 = list(mini_prob2.values())
        print(translate_name[name_of] + " & $" + str(lk2[np.argmin(lv2)]) + "$ & $" + str_convert(min(lv2) * (10 ** 5)) + "$ \\\\ \\hline")
    if p3:
        probability_of_in_next_next_step = load_object("probability/probability_of_" + name_of + "_in_next_next_step")
        mini_prob3 = dict()
        for k1 in probability_of_in_next_next_step:
            if k1 == "undefined":
                continue
            for k2 in probability_of_in_next_next_step[k1]:
                if k2 == "undefined":
                    continue
                for k3 in probability_of_in_next_next_step[k1][k2]:
                    if k3 == "undefined":
                        continue
                    mini_prob3[str_convert(k1) + "$ & $" + str_convert(k2) + "$ & $" + str_convert(k3)] = probability_of_in_next_next_step[k1][k2][k3]
        lk3 = list(mini_prob3.keys())
        lv3 = list(mini_prob3.values())
        print(translate_name[name_of] + " & $" + str(lk3[np.argmin(lv3)]) + "$ & $" + str_convert(min(lv3) * (10 ** 4)) + "$ \\\\ \\hline")

print("Variable & $x_{1}$ & $P(X_{i})$ \\\\ \\hline")
for v in translate_name: 
    get_var(v, p1 = True)
print("Variable & $X_{i-1}$ & $X_{i}$ & $P(X_{i}|X_{i-1})$ \\\\ \\hline")
for v in translate_name: 
    get_var(v, p2 = True)
print("Variable & $X_{i-1}$ & $X_{i-1}$ & $X_{i}$ & $P(X_{i}|X_{i-1}, X_{i-2})$ \\\\ \\hline")
for v in translate_name: 
    get_var(v, p3 = True)