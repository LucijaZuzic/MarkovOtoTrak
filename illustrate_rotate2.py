import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rc
cm = 1/2.54  # centimeters in inches

def make_arc(cy, cx, radi, angle_min, angle_max, angle_step):
    xvals_radius = [np.cos(angle) * radi + cx for angle in np.arange(angle_min, angle_max, angle_step)]
    yvals_radius = [np.sin(angle) * radi + cy for angle in np.arange(angle_min, angle_max, angle_step)]
    return xvals_radius, yvals_radius

def make_arrow(bx, by, len_arrow, angle = np.pi / 4, isdashed = False):
    ax = bx - np.cos((angle + np.pi / 8 + 360) % 360) * len_arrow
    ay = by - np.sin((angle + np.pi / 8 + 360) % 360) * len_arrow
    if isdashed:
        plt.plot([ax, bx], [ay, by], c = "k", linestyle = "dashed")
    else:
        plt.plot([ax, bx], [ay, by], c = "k")
    ax = bx - np.cos((angle - np.pi / 8 + 360) % 360) * len_arrow
    ay = by - np.sin((angle - np.pi / 8 + 360) % 360) * len_arrow
    if isdashed:
        plt.plot([ax, bx], [ay, by], c = "k", linestyle = "dashed")
    else:
        plt.plot([ax, bx], [ay, by], c = "k")

radi_use = 5
step_val = 10 ** -2
angle_orig = np.pi / 5

plt.rcParams["svg.fonttype"] = "none"
rc('font',**{'family':'Arial'})
#plt.rcParams.update({"font.size": 5})
SMALL_SIZE = 5
MEDIUM_SIZE = 5
BIGGER_SIZE = 5

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.figure(figsize=(29.7/4*cm, 29.7/4*cm), dpi = 300)

plt.axis("equal")
plt.axis("off")
x1 = radi_use * np.cos(angle_orig)
y1 = radi_use * np.sin(angle_orig)
xoff = radi_use / 6
yoff = radi_use / 6
plt.plot([xoff, xoff + x1], [yoff, yoff + y1], c = "k")
plt.plot([xoff, xoff], [0, 2 * yoff + radi_use], c = "k")
make_arrow(xoff, 2 * yoff + radi_use, radi_use / 6, angle = np.pi / 2)
plt.text(1.5 * xoff, 1.5 * yoff + radi_use, "$y$")
plt.text(1.5 * xoff + radi_use, 1.5 * yoff, "$x$")
plt.plot([0, 2 * xoff + radi_use], [yoff, yoff], c = "k")
make_arrow(2 * xoff + radi_use, yoff, radi_use / 6, angle = 2 * np.pi) 
x3, y3 = make_arc(yoff, xoff, radi_use, 0, angle_orig + step_val / 5, step_val)
plt.plot(x3, y3, c = "k")
plt.text(max(y1 , x1) * 2 / 3, 1.6 * yoff, "$\\theta$")
x4, y4 = make_arc(yoff, xoff, radi_use * 5 / 6, angle_orig, np.pi / 2 + step_val / 5, step_val)
plt.plot(x4, y4, c = "k")
plt.text(1.6 * xoff, 3.5 * yoff, "$90\degree - \\theta$") 
plt.title("Subtracting the heading from $90\degree$")
if not os.path.isdir("illustrate2/"):
    os.makedirs("illustrate2")
plt.savefig("illustrate2/90.png", bbox_inches = "tight")
plt.savefig("illustrate2/90.pdf", bbox_inches = "tight")
plt.savefig("illustrate2/90.svg", bbox_inches = "tight")
plt.close()

plt.rcParams["svg.fonttype"] = "none"
rc('font',**{'family':'Arial'})
#plt.rcParams.update({"font.size": 5})
SMALL_SIZE = 5
MEDIUM_SIZE = 5
BIGGER_SIZE = 5

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.figure(figsize=(29.7/4*cm, 29.7/4*cm), dpi = 300)

plt.axis("equal")
plt.axis("off")
angle_orig = np.pi / 2 - angle_orig
x1 = radi_use * 5 / 6 * np.cos(angle_orig)
y1 = radi_use * 5 / 6 * np.sin(angle_orig)
xoff = radi_use / 6
yoff = radi_use / 6
plt.plot([xoff + radi_use, xoff + radi_use + x1], [yoff, yoff + y1], c = "k")
plt.plot([0, 2 * xoff + 2 * radi_use], [yoff, yoff], c = "k")
make_arrow(2 * xoff + 2 * radi_use, yoff, radi_use / 6, angle = 2 * np.pi)
plt.plot([xoff + radi_use, xoff + radi_use], [0, yoff * 2 + radi_use], c = "k", linestyle = "dashed")
make_arrow(xoff + radi_use, yoff * 2 + radi_use, radi_use / 6, angle = np.pi / 2, isdashed = True)
plt.text(1.5 * xoff + radi_use, 1.5 * yoff + radi_use, "$y$")
plt.text(1.5 * xoff + 2 * radi_use, 1.5 * yoff, "$x$")
x3 = radi_use * np.cos(np.pi - angle_orig)
y3 = radi_use * np.sin(np.pi - angle_orig)
plt.plot([xoff + radi_use, xoff + radi_use + x3], [yoff, yoff + y3], c = "k")
x4, y4 = make_arc(yoff, xoff + radi_use, radi_use * 5 / 6, 0, angle_orig + step_val / 5, step_val)
plt.plot(x4, y4, c = "k")
x5, y5 = make_arc(yoff, xoff + radi_use, radi_use, 0, np.pi - angle_orig + step_val / 5, step_val)
plt.plot(x5, y5, c = "k")
x6, y6 = make_arc(yoff, xoff + radi_use, radi_use * 5 / 6, np.pi - angle_orig, np.pi + step_val / 5, step_val)
plt.plot(x6, y6, c = "k")
plt.text(radi_use + max(y1 , x1) * 2 / 3, 1.7 * yoff, "$\\theta$")
plt.text(xoff + radi_use - max(y1 , x1) * 2 / 3 + xoff / 2, 1.7 * yoff, "$\\theta$")
plt.text(radi_use - 2.2 * xoff, yoff + 0.75 * radi_use, "$180\degree - \\theta$") 
x7, y7 = make_arc(yoff, xoff + radi_use, radi_use / 2, angle_orig / 2, np.pi - angle_orig / 2, step_val)
plt.plot(x7, y7, c = "k", linestyle = "dashed")
make_arrow(x7[0], y7[0], radi_use / 6, angle = np.pi * 27 / 16, isdashed = True)
plt.title("Subtracting the heading from $180\degree$\n when mirroring around the $y$ axis")
if not os.path.isdir("illustrate2/"):
    os.makedirs("illustrate2")
plt.savefig("illustrate2/180.png", bbox_inches = "tight")
plt.savefig("illustrate2/180.pdf", bbox_inches = "tight")
plt.savefig("illustrate2/180.svg", bbox_inches = "tight")
plt.close()
  
plt.rcParams["svg.fonttype"] = "none"
rc('font',**{'family':'Arial'})
#plt.rcParams.update({"font.size": 5})
SMALL_SIZE = 5
MEDIUM_SIZE = 5
BIGGER_SIZE = 5

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.figure(figsize=(29.7/4*cm, 29.7/4*cm), dpi = 300)

plt.axis("equal")
plt.axis("off")
angle_orig = np.pi - angle_orig
x1 = radi_use * 5 / 6 * np.cos(angle_orig)
y1 = radi_use * 5 / 6 * np.sin(angle_orig)
xoff = radi_use / 6
yoff = radi_use / 6
plt.plot([xoff + radi_use, xoff + radi_use + x1], [yoff + radi_use, yoff + radi_use + y1], c = "k")
plt.plot([0, xoff + radi_use], [yoff + radi_use, yoff + radi_use], c = "k", linestyle = "dashed")
plt.plot([xoff + 2 * radi_use, 2 * xoff + 2 * radi_use], [yoff + radi_use, yoff + radi_use], c = "k", linestyle = "dashed")
plt.plot([xoff + radi_use, xoff + 2 * radi_use], [yoff + radi_use, yoff + radi_use], c = "k")
make_arrow(2 * xoff + 2 * radi_use, yoff + radi_use, radi_use / 6, angle = 2 * np.pi, isdashed = True)
plt.text(1.5 * xoff + 2 * radi_use, 1.5 * yoff + radi_use, "$x$")
x3 = radi_use * np.cos(2 * np.pi - angle_orig)
y3 = radi_use * np.sin(2 * np.pi - angle_orig)
plt.plot([xoff + radi_use, xoff + radi_use + x3], [yoff + radi_use, yoff + radi_use + y3], c = "k")
x4, y4 = make_arc(yoff + radi_use, xoff + radi_use, radi_use * 5 / 6, 0, angle_orig + step_val / 5, step_val)
plt.plot(x4, y4, c = "k")
x5, y5 = make_arc(yoff + radi_use, xoff + radi_use, radi_use, 0, 2 * np.pi - angle_orig + step_val / 5, step_val)
plt.plot(x5, y5, c = "k")
x6, y6 = make_arc(yoff + radi_use, xoff + radi_use, radi_use * 5 / 6, 2 * np.pi - angle_orig, 2 * np.pi + step_val / 5, step_val)
plt.plot(x6, y6, c = "k")
plt.text(radi_use + 1.2 * xoff, radi_use * 3 / 4 + yoff, "$\\theta$")
plt.text(radi_use + 1.2 * xoff, radi_use * 7 / 4 - yoff * 2.3, "$\\theta$")
plt.text(radi_use / 3.9, 3 * yoff + radi_use, "$360\degree - \\theta$")
x7, y7 = make_arc(yoff + radi_use, xoff + radi_use, radi_use / 2, angle_orig / 2, 2 * np.pi - angle_orig / 2, step_val)
plt.plot(x7, y7, c = "k", linestyle = "dashed")
make_arrow(x7[0], y7[0], radi_use / 6, angle = np.pi * 15 / 8, isdashed = True)
plt.title("Subtracting the heading from $360\degree$\n when mirroring around the $x$ axis")
if not os.path.isdir("illustrate2/"):
    os.makedirs("illustrate2")
plt.savefig("illustrate2/360.png", bbox_inches = "tight")
plt.savefig("illustrate2/360.svg", bbox_inches = "tight")
plt.savefig("illustrate2/360.pdf", bbox_inches = "tight")
plt.close()