from utilities import load_object 
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from example_distribution import get_bins, get_bins_simple, summarize_dict, summarize_2d_dict, summarize_3d_dict
    
translate_prob = {
             "direction": "Heading ($\degree$)",  
             "latitude_no_abs": "y offset ($\degree$ lat.)",  
             "longitude_no_abs": "x offset ($\degree$ long.)",   
             "time": "Time (s)",
             "speed": "Speed (km/h)", 
             }

keys_name_list = ["_simple", "_p1", "_p2", "_p3"]

def translate_title(title, type = 0):
    title_new = title.replace("predicted_", "")
    for tp in translate_prob:
        title_new = title_new.replace(tp, translate_prob[tp])
    for kn in keys_name_list:
        title_new = title_new.replace(kn, "")
    if type == 1:
        title_new = title_new.replace("_", "\nThe previous state: ")
    if type == 2:
        title_new = title_new.replace("_", "\nThe state before the previous state: ")
    if type == 3:
        ix_replace = title_new.find("_")
        title_new = title_new[:ix_replace] + "\nThe state before the previous state: " + title_new[ix_replace + 1:].replace("_", "\nThe previous state: ")
    return title_new

def dict_to_pie(dicti, title, type):
    
    if not os.path.isdir("pie"):
        os.makedirs("pie")

    plt.figure(figsize = (5, 5), dpi = 80)
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.title(translate_title(title, type))
    vals = list(dicti.values())
    labs = list(dicti.keys())
    labs = ["$" + lab + "$" for lab in labs]
    plt.pie(x = vals, labels = labs, autopct = '%1.2f%%')
    plt.savefig("pie/" + title + ".png", bbox_inches = "tight")
    plt.close()

def dict_to_bar(dicti, title, type):
    
    if not os.path.isdir("bar"):
        os.makedirs("bar")

    xarr = list(dicti.keys())
    undef = "undefined" in xarr
    if undef:
        xarr.remove("undefined")
    xarr = sorted(xarr)
    if undef:
        xarr.append("undefined")
    sum_dicti = sum(list(dicti.values()))
    yarr = [dicti[x] / sum_dicti * 100 for x in xarr]

    tick_pos = [i for i in range(0, len(xarr), max(1, len(xarr) // 5))]
    tick_labels = ["$" + str(xarr[i]).replace("undefined", "") + "$" for i in tick_pos]
    
    mul_fact = 3
    if "direction" in title or "speed" in title:
        mul_fact = 1
    plt.figure(figsize = (5 * mul_fact, 2 * mul_fact), dpi = 80)
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.title(translate_title(title, type))
    plt.bar(range(len(yarr)), yarr)
    for i in range(len(yarr)):
        plt.text(i, yarr[i] + 3 / mul_fact, str(np.round(yarr[i], 2)))
    plt.ylim(0, max(yarr) + 30 / mul_fact)
    plt.xticks(tick_pos, tick_labels)
    plt.xlabel("Next state")
    plt.ylabel("Probability (%)")
    plt.savefig("bar/" + title + ".png", bbox_inches = "tight")
    plt.close()

def dict_to_heatmap(dicti, title, type):
    
    if not os.path.isdir("heat"):
        os.makedirs("heat")

    xarr = list(dicti.keys())
    undef = "undefined" in xarr
    if undef:
        xarr.remove("undefined")
    xarr = sorted(xarr)
    if undef:
        xarr.append("undefined")
    
    data_heat = []
    for x1 in xarr:
        data_heat.append([])
        for x2 in xarr:
            if x1 in dicti and x2 in dicti[x1]:
                data_heat[-1].append(dicti[x1][x2])
            else:
                data_heat[-1].append(10 ** -20)

    xlabels = ["$" + x.replace("undefined", "") + "$" for x in xarr]

    data_heat = pd.DataFrame(data_heat, columns = xlabels, index = xlabels)

    mul_fact = 3
    if "direction" in title or "speed" in title:
        mul_fact = 1
    plt.figure(figsize = (5 * mul_fact, 5 * mul_fact), dpi = 80)
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    ax = sns.heatmap(data_heat, annot = True, fmt = '.2f')
    for t in ax.texts: t.set_text(t.get_text() + " %")
    plt.title(translate_title(title, type))
    plt.xlabel("Next state")
    plt.ylabel("Previous state")
    plt.savefig("heat/" + title + ".png", bbox_inches = "tight")
    plt.close()
     
def dict_to_heatmap2d(dicti, title, type):
    
    if not os.path.isdir("heat2d"):
        os.makedirs("heat2d")

    xarr = list(dicti.keys())
    undef = "undefined" in xarr
    if undef:
        xarr.remove("undefined")
    xarr = sorted(xarr)
    if undef:
        xarr.append("undefined")

    xarr2d = []
    
    data_heat = []
    for x1 in xarr:
        for x2 in xarr:
            data_heat.append([])
            xarr2d.append("[" + str(x1) + ", " + str(x2) + "]")
            for x3 in xarr:
                if x1 in dicti and  x2 in dicti[x1] and x3 in dicti[x1][x2]:
                    data_heat[-1].append(dicti[x1][x2][x3])
                else:
                    data_heat[-1].append(10 ** -20)

    ylabels = ["$" + x.replace("undefined", "") + "$" for x in xarr2d] 

    xlabels = ["$" + x.replace("undefined", "") + "$" for x in xarr] 

    data_heat = pd.DataFrame(data_heat, columns = xlabels, index = ylabels)

    mul_fact = 3
    if "direction" in title or "speed" in title:
        mul_fact = 1
    plt.figure(figsize = (5 * mul_fact, 5 * mul_fact), dpi = 80)
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    ax = sns.heatmap(data_heat, annot = True, fmt = '.2f')
    for t in ax.texts: t.set_text(t.get_text() + " %")
    plt.title(translate_title(title, type))
    plt.xlabel("Next state")
    plt.ylabel("Previous two states")
    plt.savefig("heat2d/" + title + ".png", bbox_inches = "tight")
    plt.close()

name_of_var = os.listdir("predicted")
nbins = 2
for name_of in name_of_var:  
    print(name_of) 

    probability_of_in_next_next_step = load_object("probability/probability_of_" + name_of.replace("predicted_", "") + "_in_next_next_step")   
    probability_of_in_next_step = load_object("probability/probability_of_" + name_of.replace("predicted_", "") + "_in_next_step")   
    probability_of = load_object("probability/probability_of_" + name_of.replace("predicted_", ""))

    print(len(probability_of_in_next_next_step))
    print(len(probability_of_in_next_step))
    print(len(probability_of))

    keys_list = list(probability_of.keys())
    if "undefined" in keys_list:
        keys_list.remove("undefined") 
    keys_list = sorted(keys_list)
    keys_list2 = list(probability_of_in_next_step.keys())
    if "undefined" in keys_list2:
        keys_list2.remove("undefined") 
    keys_list2 = sorted(keys_list2)
    keys_list3 = list(probability_of_in_next_next_step.keys())
    if "undefined" in keys_list3:
        keys_list3.remove("undefined") 
    keys_list3 = sorted(keys_list3)

    total2 = sum([sum(probability_of_in_next_step[x].values()) for x in probability_of_in_next_step])
    total3 = sum([sum(probability_of_in_next_next_step[x][y].values()) for x in probability_of_in_next_next_step for y in probability_of_in_next_next_step[x]])
    probof2 = {x: sum(probability_of_in_next_step[x].values()) / total2 for x in probability_of_in_next_step}
    probof3 = {x: sum(probability_of_in_next_next_step[x][y].values()) / total3 for x in probability_of_in_next_next_step for y in probability_of_in_next_next_step[x]}
  
    keys_new0 = get_bins_simple(keys_list, nbins)
    keys_new1 = get_bins(keys_list, probability_of, nbins)
    keys_new2 = get_bins(keys_list2, probof2, nbins)
    keys_new3 = get_bins(keys_list3, probof3, nbins) 

    keys_name = {"_simple": keys_new0, "_p1": keys_new1, "_p2": keys_new2, "_p3": keys_new3}
    
    for keys_new_name in keys_name:

        keys_new = keys_name[keys_new_name]

        n1 = summarize_dict(probability_of, keys_new, max(keys_list))
        n2 = summarize_2d_dict(probability_of_in_next_step, keys_new, max(keys_list))
        n3 = summarize_3d_dict(probability_of_in_next_next_step, keys_new, max(keys_list))
         
        dict_to_bar(n1, name_of + keys_new_name, 0)
        dict_to_pie(n1, name_of + keys_new_name, 0)

        #for lab in n2:
            #dict_to_bar(n2[lab], name_of + keys_new_name + "_" + str(lab), 1)
            #dict_to_pie(n2[lab], name_of + keys_new_name + "_" + str(lab), 1)

        dict_to_heatmap(n2, name_of + keys_new_name, 0)
        dict_to_heatmap2d(n3, name_of + keys_new_name, 0)
    
        #for lab1 in n3:
            #dict_to_heatmap(n3[lab1], name_of + keys_new_name + "_" + str(lab1), 2)
            #for lab2 in n3[lab1]:
                #dict_to_bar(n3[lab1][lab2], name_of + keys_new_name + "_" + str(lab1) + "_" + str(lab2), 3)
                #dict_to_pie(n3[lab1][lab2], name_of + keys_new_name + "_" + str(lab1) + "_" + str(lab2), 3)