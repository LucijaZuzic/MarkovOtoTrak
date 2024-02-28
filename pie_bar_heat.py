from utilities import load_object 
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
    
translate_prob = {
             "direction": "Heading",  
             "latitude_no_abs": "y offset",  
             "longitude_no_abs": "x offset",   
             "time": "Time (s)",
             "speed": "Speed (km/h)", 
             }

def translate_title(title, type = 0):
    title_new = title.replace("predicted_", "")
    for tp in translate_prob:
        title_new = title_new.replace(tp, translate_prob[tp])
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

    plt.figure(figsize = (15, 15), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.title(translate_title(title, type))
    plt.pie(list(dicti.values()), labels = list(dicti.keys()), autopct = '%1.2f%%', pctdistance = 1.1, labeldistance = 1.2)
    plt.savefig("pie/" + title + ".png", bbox_inches = "tight")
    plt.close()

def dict_to_bar(dicti, title, type):
    
    if not os.path.isdir("bar"):
        os.makedirs("bar")

    xarr = list(dicti.keys())
    xarr.remove("undefined")
    xarr = sorted(xarr)
    xarr.append("undefined")
    yarr = [dicti[x] * 100 for x in xarr]

    tick_pos = [i for i in range(0, len(xarr), len(xarr) // 5)]
    tick_labels = [str(xarr[i]).replace("undefined", "") for i in tick_pos]
 
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.title(translate_title(title, type))
    plt.bar(range(len(yarr)), yarr)
    plt.xticks(tick_pos, tick_labels)
    plt.xlabel("Next state")
    plt.ylabel("Probability (%)")
    plt.savefig("bar/" + title + ".png", bbox_inches = "tight")
    plt.close()

def dict_to_heatmap(dicti, title, type):
    
    if not os.path.isdir("heat"):
        os.makedirs("heat")

    xarr = list(dicti.keys())
    xarr.remove("undefined")
    xarr = sorted(xarr)
    xarr.append("undefined")
    
    data_heat = []
    for x1 in xarr:
        data_heat.append([])
        for x2 in xarr:
            if x1 in dicti and x2 in dicti[x1]:
                data_heat[-1].append(dicti[x1][x2])
            else:
                data_heat[-1].append(10 ** -20)

    xlabels = [x for x in xarr]
    xlabels[-1] = ""

    data_heat = pd.DataFrame(data_heat, columns = xlabels, index = xlabels)

    plt.figure(figsize = (15, 15), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    sns.heatmap(data_heat) 
    plt.title(translate_title(title, type))
    plt.xlabel("Next state")
    plt.ylabel("Previous state")
    plt.savefig("heat/" + title + ".png", bbox_inches = "tight")
    plt.close()
     
def dict_to_heatmap2d(dicti, title, type):
    
    if not os.path.isdir("heat2d"):
        os.makedirs("heat2d")

    xarr = list(dicti.keys())
    xarr.remove("undefined")
    xarr = sorted(xarr)
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

    ylabels = [x.replace("undefined", "") for x in xarr2d]
    ylabels[-1] = ""

    xlabels = [x for x in xarr]
    xlabels[-1] = ""

    data_heat = pd.DataFrame(data_heat, columns = xlabels, index = ylabels)

    plt.figure(figsize = (15, 15), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    sns.heatmap(data_heat) 
    plt.title(translate_title(title, type))
    plt.xlabel("Next state")
    plt.ylabel("Previous two states")
    plt.savefig("heat2d/" + title + ".png", bbox_inches = "tight")
    plt.close()

name_of_var = os.listdir("predicted")
for name_of in name_of_var:  
    print(name_of) 

    probability_of_in_next_next_step = load_object("probability/probability_of_" + name_of.replace("predicted_", "") + "_in_next_next_step")   
    probability_of_in_next_step = load_object("probability/probability_of_" + name_of.replace("predicted_", "") + "_in_next_step")   
    probability_of = load_object("probability/probability_of_" + name_of.replace("predicted_", ""))

    print(len(probability_of_in_next_next_step))
    print(len(probability_of_in_next_step))
    print(len(probability_of))

    dict_to_bar(probability_of, name_of, 0)
    #dict_to_pie(probability_of, name_of, 0)

    #for lab in probability_of_in_next_step:
        #dict_to_bar(probability_of_in_next_step[lab], name_of + "_" + str(lab), 1)
        #dict_to_pie(probability_of_in_next_step[lab], name_of + "_" + str(lab), 1)

    dict_to_heatmap(probability_of_in_next_step, name_of, 0)
    dict_to_heatmap2d(probability_of_in_next_next_step, name_of, 0)
  
    #for lab1 in probability_of_in_next_next_step:
        #dict_to_heatmap(probability_of_in_next_next_step[lab1], name_of + "_" + str(lab1), 2)
        #for lab2 in probability_of_in_next_next_step[lab1]:
            #dict_to_bar(probability_of_in_next_next_step[lab1][lab2], name_of + "_" + str(lab1) + "_" + str(lab2), 3)
            #dict_to_pie(probability_of_in_next_next_step[lab1][lab2], name_of + "_" + str(lab1) + "_" + str(lab2), 3)