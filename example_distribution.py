from utilities import * 

def save_table(str_save, name_save):
    if not os.path.isdir("distribution"):
        os.makedirs("distribution")
    file_w = open("distribution/" + name_save + ".txt", "w")
    file_w.write(str_save)
    file_w.close()

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

def header_dict(dictio):
    str_pr = ""
    for k in dictio:
        str_pr += "$" + str(k) + "$ & "
    str_pr = str_pr[:-3]
    str_pr += "\\\\ \\hline\n"
    return str_pr
 
def print_1d(dictio, mul, name_save):
    str_pr = "\\begin{table}\n\\centering\n"
    str_pr += "\\begin{tabular}{|" + "c|" * (len(dictio) + 1) + "}\n\\hline\n"
    str_pr += "$X_{i}$ & " + header_dict(dictio) + "$P(X_{i})$ & "
    for k in dictio:
        str_pr += "$" + str(np.round(dictio[k] * mul, 2)) + "\%$ & "
    str_pr = str_pr[:-3]
    str_pr += "\\\\ \\hline\n\\end{tabular}\n"
    str_pr_short = str_pr.replace("\\begin{table}\n\\centering\n", "")
    str_pr_shortest = str_pr_short.replace("\\begin{tabular}{|" + "c|" * (len(dictio) + 1) + "}\n", "")
    str_pr_shortest = str_pr_shortest.replace("\\end{tabular}\n", "")
    str_pr += "\\caption{" + name_save.replace("_", " ").capitalize() + "}\n"
    str_pr += "\\label{tab:" + name_save + "}\n"
    str_pr += "\\end{table}\n"
    if not name_save == "":
        save_table(str_pr, name_save)
    return str_pr, str_pr_short, str_pr_shortest

def print_2d(dictio, mul, name_save = ""):
    str_pr = "\\begin{table}\n\\centering\n"
    str_pr += "\\begin{tabular}{|" + "c|" * (len(dictio) + 1) + "}\n\\hline\n"
    str_pr += "$P(X_{i}|X_{i-1})$ & \\multicolumn{" + str(len(dictio)) + "}{|c|}{$X_{i}$}\\\\ \\hline\n"
    str_pr += "$X_{i-1}$ & "  
    ix = 0
    for prev in dictio: 
        ix += 1
        str_pr += "$V_{" + str(ix) + "}$ & "
    str_pr = str_pr[:-3]
    str_pr += "\\\\ \\hline\n"
    ix = 0 
    for prev in dictio: 
        ix += 1
        str_pr += "$V_{" + str(ix) + "}=" + str(prev) + "$ & "
        for k in dictio:
            if k in dictio[prev]:
                str_pr += "$" + str(np.round(dictio[prev][k] * mul, 2)) + "\\%$ & "
            else:
                str_pr += "$" + str(np.round(0, 2)) + "\\%$ & "
        str_pr = str_pr[:-3]
        str_pr += "\\\\ \\hline\n"
    str_pr += "\\end{tabular}\n"
    str_pr_short = str_pr.replace("\\begin{table}\n\\centering\n", "")
    str_pr_shortest = str_pr_short.replace("\\begin{tabular}{|" + "c|" * (len(dictio) + 1) + "}\n", "")
    str_pr_shortest = str_pr_shortest.replace("\\end{tabular}\n", "")
    str_pr += "\\caption{" + name_save.replace("_", " ").capitalize() + "}\n"
    str_pr += "\\label{tab:" + name_save + "}\n"
    str_pr += "\\end{table}\n"
    if not name_save == "":
        save_table(str_pr, name_save)
    return str_pr, str_pr_short, str_pr_shortest

def print_3d(dictio, mul, name_save): 
    str_pr = "\\begin{table}\n\\centering\n"
    str_pr += "\\begin{tabular}{|" + "c|" * (len(dictio) ** 2 + 1) + "}\n\\hline\n"
    str_pr += "\multirow{3}{*}{$P(X_{i}|X_{i-1},X_{i-2})$} & \\multicolumn{" + str(len(dictio) ** 2) + "}{|c|}{$X_{i-2}$}\\\\ \\cline{2-" + str(len(dictio) ** 2 + 1) + "}\n"
    str_pr += " & "
    ix = 0
    for prev in dictio: 
        ix += 1
        str_pr += "\\multicolumn{" + str(len(dictio)) + "}{|c|}{$V_{" + str(ix) + "}$} & "
    str_pr = str_pr[:-3]
    str_pr += "\\\\ \\cline{2-" + str(len(dictio) ** 2 + 1) + "}\n"
    part = "\\multicolumn{" + str(len(dictio)) + "}{|c|}{$X_{i}$}"
    str_pr += " & " + (part + " & ") * (len(dictio) - 1) + part + "\\\\ \\hline\n"
    hd_short = ""
    ix = 0
    for prev in dictio: 
        ix += 1
        hd_short += "$V_{" + str(ix) + "}$ & "
    hd = hd_short
    hd = hd[:-3]
    hd += "\\\\ \\hline\n"
    str_pr += "$X_{i-1}$ & " + hd_short * (len(dictio) - 1) + hd 
    ix = 0 
    for prev in dictio: 
        ix += 1
        str_pr += "$V_{" + str(ix) + "}=" + str(prev) + "$ & "
        for prevprev in dictio:
            for curr in dictio:
                if prev in dictio[prevprev] and curr in dictio[prevprev][prev]:
                    str_pr += "$" + str(np.round(dictio[prevprev][prev][curr] * mul, 2)) + "\\%$ & "
                else: 
                    closest_prev = ""
                    for cp in dictio:
                        if cp in dictio[prevprev] and curr in dictio[prevprev][cp]:
                            closest_prev = cp
                    if closest_prev == "":
                        str_pr += "$" + str(np.round(0, 2)) + "\\%$ & "
                    else:
                        str_pr += "$" + str(np.round(dictio[prevprev][closest_prev][curr] * mul, 2)) + "\\%$ & "
        str_pr = str_pr[:-3]
        str_pr += "\\\\ \\hline\n"
    str_pr += "\\end{tabular}\n"
    str_pr_short = str_pr.replace("\\begin{table}\n\\centering\n", "")
    str_pr_shortest = str_pr_short.replace("\\begin{tabular}{|" + "c|" * (len(dictio) + 1) + "}\n", "")
    str_pr_shortest = str_pr_shortest.replace("\\end{tabular}\n", "")
    str_pr += "\\caption{" + name_save.replace("_", " ").capitalize() + "}\n"
    str_pr += "\\label{tab:" + name_save + "}\n"
    str_pr += "\\end{table}\n"
    if not name_save == "":
        save_table(str_pr, name_save)
    return str_pr, str_pr_short, str_pr_shortest

def convert_keys(new_bins, maxval, i):
        new_min = new_bins[i]
        new_max = maxval
        if i + 1 < len(new_bins):
            new_max = new_bins[i + 1]
        new_key = "[" + str_convert(new_min) + ", " + str_convert(new_max)
        if i + 1 < len(new_bins):
            new_key += ">"
        else:
            new_key += "]"
        return new_min, new_max, new_key

def summarize_dict(old_dict, new_bins, maxval): 
    new_dict = dict()
    for i in range(len(new_bins)): 
        new_min, new_max, new_key = convert_keys(new_bins, maxval, i)
        new_dict[new_key] = 0
        for k in old_dict:
            if k == "undefined":
                continue
            if k >= new_min and k < new_max:
                new_dict[new_key] += old_dict[k]
    for k in new_dict:   
        if new_dict[k] >= 0.9999:
            new_dict[k] = 0.9999
        if new_dict[k] <= 0.0001:
            new_dict[k] = 0.0001
        new_dict[k] = np.round(new_dict[k] * 100, 2)  
    return new_dict

def summarize_2d_dict(old_dict, new_bins, maxval): 
    new_1d_fixed = dict()
    for k in old_dict:
        new_1d_fixed[k] = summarize_dict(old_dict[k], new_bins, maxval)  
    new_dict = dict()
    num_in_old_dict = dict()
    for i in range(len(new_bins)):
        new_min, new_max, new_key = convert_keys(new_bins, maxval, i)
        new_dict[new_key] = dict()
        num_in_old_dict[new_key] = 0 
        for k in old_dict:
            if k == "undefined":
                continue 
            if k >= new_min and k < new_max:
                num_in_old_dict[new_key] += 1  
                for new_key2 in new_1d_fixed[k]:
                    if new_key2 not in new_dict[new_key]:
                        new_dict[new_key][new_key2] = 0 
                    new_dict[new_key][new_key2] += new_1d_fixed[k][new_key2]  
    for k in new_dict:
        for k2 in new_dict[k]:
            new_dict[k][k2] /= num_in_old_dict[k]    
            new_dict[k][k2] = np.round(new_dict[k][k2], 2)  
    return new_dict

def summarize_3d_dict(old_dict, new_bins, maxval): 
    new_2d_fixed = dict()
    for k in old_dict:
        new_2d_fixed[k] = summarize_2d_dict(old_dict[k], new_bins, maxval)  
    new_dict = dict()
    for i in range(len(new_bins)):
        new_min, new_max, new_key = convert_keys(new_bins, maxval, i)
        new_dict[new_key] = dict() 
        for k in old_dict:
            if k == "undefined":
                continue 
            if k >= new_min and k < new_max:
                for new_key2 in new_2d_fixed[k]:
                    if new_key2 not in new_dict[new_key]:
                        new_dict[new_key][new_key2] = dict() 
                        for new_key3 in new_2d_fixed[k][new_key2]:
                            if new_key3 not in new_dict[new_key][new_key2]:
                                new_dict[new_key][new_key2][new_key3] = 0 
                            new_dict[new_key][new_key2][new_key3] += new_2d_fixed[k][new_key2][new_key3]
    for k in new_dict:
        for k2 in new_dict[k]:
            for k3 in new_dict[k][k2]:  
                new_dict[k][k2][k3] = np.round(new_dict[k][k2][k3], 2)  
    return new_dict

def get_bins_simple(keys_list, num_bins): 
    minr = min(keys_list)
    maxr = max(keys_list)
    if maxr > 358:
        maxr = 360
    stepr = (maxr - minr) / num_bins
    return np.arange(minr, maxr, stepr)

def get_bins(keys_list, probability_of, num_bins): 
    so_far = []
    for k in keys_list:
        so_far.append(probability_of[k]) 
    new_bins_indexes = [i for i in range(num_bins)]
    for index_working in range(num_bins - 1, 0, -1):
        curr = new_bins_indexes[index_working]
        if index_working != num_bins - 1:
            next = new_bins_indexes[index_working + 1] 
        else:
            next = -1
        prev = new_bins_indexes[index_working - 1]
        while curr > 0 and curr > prev - 1 and sum(so_far[curr:next]) > 1 / num_bins:
            curr += 1 
        new_bins_indexes[index_working] = curr
    new_bins = [keys_list[x] for x in new_bins_indexes]
    return new_bins

def get_var(name_of):
    print(name_of)
    #predicted = load_object("predicted/predicted_" + name_of)   

    probability_of_in_next_next_step = load_object("probability/probability_of_" + name_of + "_in_next_next_step")   
    probability_of_in_next_step = load_object("probability/probability_of_" + name_of + "_in_next_step")   
    probability_of = load_object("probability/probability_of_" + name_of) 
   
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
    nbins = ncols
    mul = 1
    if "sgn" in name_of: 
        n1 = probability_of
        n2 = probability_of_in_next_step
        n3 = probability_of_in_next_next_step
        mul = 100
    else:
        keys_new1 = get_bins(keys_list, probability_of, nbins)
        keys_new2 = get_bins(keys_list2, probof2, nbins)
        keys_new3 = get_bins(keys_list3, probof3, nbins)
        keys_new = keys_new1
        if name_of == "speed" or name_of == "distance":
            keys_new = keys_new3
        keys_new = get_bins_simple(keys_list, nbins)
        n1 = summarize_dict(probability_of, keys_new, max(keys_list))
        n2 = summarize_2d_dict(probability_of_in_next_step, keys_new, max(keys_list))
        n3 = summarize_3d_dict(probability_of_in_next_next_step, keys_new, max(keys_list))
    
    p1, p1s, p1ss = print_1d(n1, mul, name_of + "_1d")
    p2, p2s, p2ss = print_2d(n2, mul, name_of + "_2d")
    p3, p3s, p3ss = print_3d(n3, mul, name_of + "_3d")
    repl_name = name_of.replace("_", " ").capitalize()
    save_table("\chapter{" + repl_name + "}\n" + p1 + p2 + p3, name_of + "_all") 
    str_pr_short = "\chapter{" + repl_name + "}\n\\begin{table}\n\\centering\n" + p1s + p2s + p3s
    str_pr_short += "\\caption{" + repl_name + "}\n"
    str_pr_short += "\\label{tab:" + name_of + "}\n"
    str_pr_short += "\\end{table}\n"
    save_table(str_pr_short, name_of + "_all_short")
    print(p1)
    print(p2)
    print(p3)
    return "\chapter{" + name_of.replace("_", " ").capitalize() + "}\n" + p1 + p2 + p3, str_pr_short, p1ss, p2ss, p3ss

totally = ""
totally_short = ""
ncols = 2
shortest_p1 = "\\begin{table}\n\\centering\n"
shortest_p1 += "\\begin{tabular}{|" + "c|" * (ncols + 1) + "}\n\\hline\n"

shortest_p2 = "\\begin{table}\n\\centering\n"
shortest_p2 += "\\begin{tabular}{|" + "c|" * (ncols + 1) + "}\n\\hline\n"

shortest_p3 = "\\begin{table}\n\\centering\n"
shortest_p3 += "\\begin{tabular}{|" + "c|" * (ncols ** 2 + 1) + "}\n\\hline\n"

name_of_var = os.listdir("predicted")
for v in name_of_var: 
    if v == "predicted_time_half":
        continue
    if v == "predicted_time_ten":
        continue
    starting1 = "\\multicolumn{" + str(ncols + 1) + "}{|c|}{" + v.replace("_", " ").capitalize() + "}\\\\ "
    starting2 = "\\multicolumn{" + str(ncols + 1) + "}{|c|}{" + v.replace("_", " ").capitalize() + "}\\\\ "
    starting3 = "\\multicolumn{" + str(ncols ** 2 + 1) + "}{|c|}{" + v.replace("_", " ").capitalize() + "}\\\\ "
    t, ts, s1, s2, s3 = get_var(v.replace("predicted_", ""))
    totally += t
    totally_short += ts
    shortest_p1 += starting1 + s1
    shortest_p2 += starting2 + s2
    shortest_p3 += starting3 + s3
    
shortest_p1 += "\\end{tabular}\n"
shortest_p1 += "\\caption{1d}\n"
shortest_p1 += "\\label{tab:1d}\n"
shortest_p1 += "\\end{table}\n"

shortest_p2 += "\\end{tabular}\n"
shortest_p2 += "\\caption{2d}\n"
shortest_p2 += "\\label{tab:2d}\n"
shortest_p2 += "\\end{table}\n"

shortest_p3 += "\\end{tabular}\n"
shortest_p3 += "\\caption{3d}\n"
shortest_p3 += "\\label{tab:3d}\n"
shortest_p3 += "\\end{table}\n"

save_table(totally, "all_all")
save_table(totally_short, "all_all_short")
save_table(shortest_p1, "all_p1")
save_table(shortest_p2, "all_p2")
save_table(shortest_p3, "all_p3")