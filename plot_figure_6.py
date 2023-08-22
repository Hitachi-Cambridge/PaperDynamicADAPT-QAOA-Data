import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')


data_dir = 'output_data/noisy_data/'
min_nodes = 6
max_nodes = 10
no_nodes = [i for i in range(min_nodes, max_nodes+1)]
algo_types = {}
fontsize_axis = 24
fontsize_title= 27
fontsize_legend = 20
linewidth = 4
capsize = 3*linewidth
fontsize_axis = 23

for node in no_nodes:

    tmp_df = pd.read_csv(data_dir + str(node) + '_nodes/gw_crossovers/crossovers.csv', header=None)
    for row in range(tmp_df.shape[0]):
        algo = tmp_df.iloc[row][0]
        crossover_gate_error_prob = tmp_df.iloc[row][1]
        crossover_gate_error_prob_error = tmp_df.iloc[row][2]
        if algo not in algo_types:
            algo_types[algo] = {
                'crossovers' : [crossover_gate_error_prob],
                'errors' : [crossover_gate_error_prob_error]
            }
        else:
            algo_types[algo]['crossovers'].append(crossover_gate_error_prob)
            algo_types[algo]['errors'].append(crossover_gate_error_prob_error)


fig, ax = plt.subplots(figsize = (12,6))

import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
boxes = []
images = []
annotation_boxes = []
coord_1 = 0.045
coord_5 = 0.955
coords = [0.05 + i * (0.955-0.05)/4 for i in range(5)]

ax.hlines([5/4*(1-0.9992)], 5, 11, linestyles=['dashdot'], colors=['#77cca4'], linewidth=[3.5], label='Experimental Gate-Error')
for algo in algo_types:
    if 'dynamic' in algo:
        label = 'Dynamic ADAPT-QAOA'
        color = '#fe6000'
        linestyle='solid'
        linewidth=4
    else:
        label = 'ADAPT-QAOA'
        color = '#64a5ff'
        linestyle='dashed'
        linewidth=3
    ax.errorbar(no_nodes, algo_types[algo]['crossovers'], algo_types[algo]['errors'], label=label, capsize=capsize, capthick=linewidth/2, color=color, linestyle=linestyle, linewidth=linewidth)

ax.set_xlabel('Number of vertices, $N$', fontsize=fontsize_axis)
ax.set_ylabel('$p_\mathrm{gate}^{\star}$', fontsize=fontsize_axis)
ax.set_yscale('log')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim(5.8, 10.2)
ax.legend(loc='upper right', fontsize=fontsize_legend)
ax.tick_params(axis='both', which='major', labelsize=fontsize_axis)
ax.tick_params(axis='both', which='minor', labelsize=fontsize_axis-2)

# plt.show()

for index, no_nodes in enumerate([6,7,8,9,10]):

    images.append(plt.imread('figures/' + str(no_nodes) + '_nodes.png'))
    boxes.append(OffsetImage(images[index], zoom = 0.2))
    annotation_boxes.append(AnnotationBbox(boxes[index], (coords[index],1.2), frameon = False, xycoords='axes fraction'))
    ax.add_artist(annotation_boxes[index])

fig.savefig('figures/figure_6.pdf', transparent=False, format="pdf", bbox_inches="tight")
