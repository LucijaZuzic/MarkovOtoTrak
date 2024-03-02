import matplotlib.pyplot as plt
import numpy as np
import os
from example_distribution import get_bins, get_bins_simple, summarize_dict, summarize_2d_dict, summarize_3d_dict
from utilities import load_object

radi_use = 5
theta_use = np.pi / 4
step_val = 10 ** -2

translate_prob = {
             "direction": "Heading ($\degree$)",  
             "latitude_no_abs": "y offset ($\degree$ lat.)",  
             "longitude_no_abs": "x offset ($\degree$ long.)",   
             "time": "Time (s)",
             "speed": "Speed (km/h)", 
             }

keys_name_list = ["_simple", "_p1", "_p2", "_p3"]

def translate_savename(savename):
    title_new = savename.replace("predicted_", "")
    title_new = title_new.replace("_c2", "")
    title_new = title_new.replace("_c3", "")
    for tp in translate_prob:
        title_new = title_new.replace(tp, translate_prob[tp])
    for kn in keys_name_list:
        title_new = title_new.replace(kn, "")
    return title_new

def make_arc(cy, cx, radi, angle_min, angle_max, angle_step):
    xvals_radius = [np.cos(angle) * radi + cx for angle in np.arange(angle_min, angle_max, angle_step)]
    yvals_radius = [np.sin(angle) * radi + cy for angle in np.arange(angle_min, angle_max, angle_step)]
    return xvals_radius, yvals_radius

def make_arrow(bx, by, len_arrow, angle = np.pi / 4):
    ax = bx - np.cos((angle + np.pi / 8 + 360) % 360) * len_arrow
    ay = by - np.sin((angle + np.pi / 8 + 360) % 360) * len_arrow
    plt.plot([ax, bx], [ay, by], c = "k")
    ax = bx - np.cos((angle - np.pi / 8 + 360) % 360) * len_arrow
    ay = by - np.sin((angle - np.pi / 8 + 360) % 360) * len_arrow
    plt.plot([ax, bx], [ay, by], c = "k")

def make_circles(radi_circ, theta_angle, step_size, y0, num1, num2, num3, num4, v1, v2, begin_str, useleft = True, useright = True, uselegned = True):
    xoff = radi_circ / 6
    yoff = xoff / 4
    dist_circ = abs(2 * radi_circ * np.cos(np.pi - theta_angle))
    if useleft:
        x1, y1 = make_arc(y0 + radi_circ, radi_circ, radi_circ, theta_angle, 2 * np.pi - theta_angle, step_size)
        plt.plot(x1, y1, c = "k")
        make_arrow(x1[-1], y1[-1], xoff, 1 / 4 * np.pi)
    if useleft:
        plt.text(xoff, y0 + radi_circ - yoff, str(np.round(num1, 2)) + "%")
    else:
        plt.text(dist_circ - xoff * len(str(np.round(num1, 2))) / 2, y0 - radi_circ * 2 + yoff * 5, str(np.round(num1, 2)) + "%")
    x2, y2 = make_arc(y0 + radi_circ, radi_circ + dist_circ, radi_circ, 0, 2 * np.pi, step_size)
    plt.plot(x2, y2, c = "k")
    plt.text(radi_circ + dist_circ - xoff * (len(begin_str) / 2 + 1), y0 + radi_circ - yoff, begin_str + "V1")
    x3, y3 = make_arc(y0 + radi_circ, radi_circ + dist_circ * 2, radi_circ, theta_angle, np.pi - theta_angle, step_size)
    if useright:
        plt.plot(x3, y3, c = "k")
        make_arrow(x3[-1], y3[-1], xoff, 5 / 4 * np.pi)
    if useright:
        plt.text(radi_circ + dist_circ * 2 - xoff * len(str(np.round(num3, 2))) / 2, y0 + radi_circ * 2 + yoff * 2, str(np.round(num3, 2)) + "%")
    else:
        plt.text(radi_circ + dist_circ * 2 + xoff, y0 + radi_circ * 4 - yoff * 10, str(np.round(num3, 2)) + "%")
    x4, y4 = make_arc(y0 + radi_circ, radi_circ + dist_circ * 2, radi_circ, np.pi + theta_angle, 2 * np.pi - theta_angle, step_size)
    if useleft:
        plt.plot(x4, y4, c = "k")
        make_arrow(x4[-1], y4[-1], xoff, 1 / 4 * np.pi)
    if useleft:
        plt.text(radi_circ + dist_circ * 2 - xoff * len(str(np.round(num2, 2))) / 2, y0 - yoff * 8, str(np.round(num2, 2)) + "%")
    else:
        plt.text(radi_circ + dist_circ + xoff, y0 - radi_circ * 2 + yoff * 5, str(np.round(num2, 2)) + "%")
    if uselegned:
        plt.text(radi_circ + dist_circ - xoff, y0 - yoff * 16, "V1 = $" + v1 + "$")
        plt.text(radi_circ + dist_circ - xoff, y0 - yoff * 24, "V2 = $" + v2 + "$")
    x5, y5 = make_arc(y0 + radi_circ, radi_circ + dist_circ * 3, radi_circ, 0, 2 * np.pi, step_size)
    plt.plot(x5, y5, c = "k")
    plt.text(radi_circ + dist_circ * 3 - xoff * (len(begin_str) / 2 + 1), y0 + radi_circ - yoff, begin_str + "V2")
    if useright:
        x6, y6 = make_arc(y0 + radi_circ, radi_circ + dist_circ * 4, radi_circ, 0, np.pi - theta_angle, step_size)
        plt.plot(x6, y6, c = "k")
        x7, y7 = make_arc(y0 + radi_circ, radi_circ + dist_circ * 4, radi_circ, np.pi + theta_angle, 2 * np.pi, step_size)
        plt.plot(x7, y7, c = "k")
        make_arrow(x7[0], y7[0], xoff, 3 / 4 * np.pi)
    if useright:
        plt.text(radi_circ * 2 + dist_circ * 3 + xoff, y0 + radi_circ - yoff, str(np.round(num4, 2)) + "%")
    else:
        plt.text(radi_circ + dist_circ * 3 + xoff, y0 + radi_circ * 4 - yoff * 10, str(np.round(num4, 2)) + "%")
    if useright:
        return radi_circ + dist_circ, y0, x4[0], y4[0], dist_circ + radi_circ * 2, y0 + radi_circ, radi_circ + dist_circ * 3, y0
    return radi_circ + dist_circ, y0 + radi_circ * 2, dist_circ * 3, y0 + radi_circ, x3[0], y3[0], radi_circ + dist_circ * 3, y0 + radi_circ * 2
    
def make_two_circles(radi_circ, theta_angle, step_size, y0, offset, num1, num2, num3, num4, num5, num6, num7, num8, v1, v2, savename = ""):
    plt.figure(figsize=(10, 9), dpi = 80)
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal") 
    plt.axis("off") 
    x1, y1, x2, y2, x3, y3, x4, y4 = make_circles(radi_circ, theta_angle, step_size, y0, num1, num2, num3, num4, v1, v2, "V1, ", True, False, True)
    x5, y5, x6, y6, x7, y7, x8, y8 = make_circles(radi_circ, theta_angle, step_size, y0 + radi_circ * 2 + offset, num5, num6, num7, num8, v1, v2, "V2, ", False, True, False)
    plt.plot([x2, x6], [y2, y6], c = "k")
    plt.plot([x3, x7], [y3, y7], c = "k")
    plt.plot([x1, x5], [y1, y5], c = "k")
    plt.plot([x4, x8], [y4, y8], c = "k")
    make_arrow(x1, y1, radi_circ / 6, 3 / 2 * np.pi)
    make_arrow(x2, y2, radi_circ / 6, 13 / 8 * np.pi)
    make_arrow(x7, y7, radi_circ / 6, 5 / 8 * np.pi)
    make_arrow(x8, y8, radi_circ / 6, 1 / 2 * np.pi)
    plt.title(translate_savename(savename))
    if not os.path.isdir("graph_visual/"):
        os.makedirs("graph_visual/")
    plt.savefig("graph_visual/" + savename + ".png", bbox_inches = "tight")
    plt.close()

def make_one_circle(radi_circ, theta_angle, step_size, y0, num1, num2, num3, num4, v1, v2, useleft = True, useright = True, uselegned = True, savename = ""):
    plt.figure(figsize=(10, 4), dpi = 80)
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal") 
    plt.axis("off") 
    make_circles(radi_circ, theta_angle, step_size, y0, num1, num2, num3, num4, v1, v2, "", useleft, useright, uselegned)
    plt.title(translate_savename(savename))
    if not os.path.isdir("graph_visual/"):
        os.makedirs("graph_visual/")
    plt.savefig("graph_visual/" + savename + ".png", bbox_inches = "tight")
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

        print(keys_new_name)

        n1 = summarize_dict(probability_of, keys_new, max(keys_list))
        n2 = summarize_2d_dict(probability_of_in_next_step, keys_new, max(keys_list))
        n3 = summarize_3d_dict(probability_of_in_next_next_step, keys_new, max(keys_list))
   
        lk = list(n2.keys())
        v1 = lk[0]
        v2 = lk[1]
        if v1 not in n2[v1]:
            n2[v1][v1] = 0.01
        if v2 not in n2[v1]:
            n2[v1][v2] = 0.01
        if v1 not in n2[v2]:
            n2[v2][v1] = 0.01
        if v2 not in n2[v2]:
            n2[v2][v2] = 0.01
        num1 = n2[v1][v1]
        num2 = 100 - num1
        num3 = n2[v2][v1]
        num4 = 100 - num3
 
        lk2 = list(n3.keys())
        v3 = lk2[0]
        v4 = lk2[1]
        if v3 not in n3[v4]:
            n3[v4][v3] = dict()
        if v4 not in n3[v3]:
            n3[v3][v4] = dict()
        if v3 not in n3[v3][v3]:
            n3[v3][v3][v3] = 0.01
        if v4 not in n3[v3][v3]:
            n3[v3][v3][v4] = 0.01
        if v3 not in n3[v3][v4]:
            n3[v3][v4][v3] = 0.01
        if v4 not in n3[v3][v4]:
            n3[v3][v4][v4] = 0.01
        if v3 not in n3[v4][v3]:
            n3[v4][v3][v3] = 0.01
        if v4 not in n3[v4][v3]:
            n3[v4][v3][v4] = 0.01
        if v3 not in n3[v4][v4]:
            n3[v4][v4][v3] = 0.01
        if v4 not in n3[v4][v4]:
            n3[v4][v4][v4] = 0.01
        num5 = n3[v3][v3][v3]
        num6 = 100 - num5
        num7 = n3[v3][v4][v3]
        num8 = 100 - num7
        num9 = n3[v4][v3][v3]
        num10 = 100 - num9
        num11 = n3[v4][v4][v3]
        num12 = 100 - num11

        make_one_circle(radi_use, theta_use, step_val, radi_use, num1, num2, num3, num4, v1, v2, useleft = True, useright = True, uselegned = True, savename = name_of + keys_new_name + "_c2")
        make_two_circles(radi_use, theta_use, step_val, radi_use, radi_use * 2, num5, num6, num7, num8, num9, num10, num11, num12, v1, v2, name_of + keys_new_name + "_c3")
