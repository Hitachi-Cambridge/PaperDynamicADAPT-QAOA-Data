import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib as mpl  
from matplotlib.ticker import MaxNLocator
mpl.rc('font',family='Times New Roman')

import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

fontsize_axis = 19

output_dir = './output_data/appendices/'

fig, ax = plt.subplots(figsize=(12,8))

for dir in os.listdir(output_dir):
    
    if 'DS_Store' in dir:
        continue

    if dir == 'dynamic_adapt':
        label = 'Full Dynamic-ADAPT-QAOA'
        color = '#fe6000'
        linewidth = 4.0
    elif dir == 'dynamic_adapt_no_cost':
        label = 'No Cost Unitaries'
        color = '#64a5ff'
        linewidth=3.0
    elif dir == 'dynamic_adapt_zero_gamma':
        label = 'No Gradient Re-evaluation'
        color = '#77cca4'
        linewidth=3.0
    else:
        continue
    
    data_dir = output_dir + dir + '/' + 'convergence_data/'

    cut_approximation_ratios = []

    for file in os.listdir(data_dir):
        
        data_pd = pd.read_csv(data_dir + file)
        cut_approximation_ratios.append(data_pd['Cut Approximation Ratios'].to_list())
        
    cut_approximation_ratios = np.array(cut_approximation_ratios)
    mean_approximation_ratios = np.average(cut_approximation_ratios, axis=0)
    std_approximation_ratios = 1.96 * np.std(cut_approximation_ratios, axis=0) / np.sqrt(len(cut_approximation_ratios))
    layers = [i for i in range(len(mean_approximation_ratios))]
    ax.plot(layers, 1-mean_approximation_ratios, label=label, color=color, linewidth=linewidth)
    ax.fill_between(layers, 1-(mean_approximation_ratios-std_approximation_ratios), 1-(mean_approximation_ratios+std_approximation_ratios), alpha=0.3, color=color)
ax.legend()
ax.set_yscale('log')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis='both', which='major', labelsize=fontsize_axis-2)
ax.legend()
ax.set_yscale('log')
ax.set_ylabel(r'$1-\alpha$', fontsize=fontsize_axis)
ax.set_xlabel('Circuit Depth, $P$', fontsize=fontsize_axis)
ax.legend(loc='upper right', fontsize=15)

for index, no_nodes in enumerate([6]):

    img = plt.imread('./figures/' + str(no_nodes) + '_nodes.png')
    box = OffsetImage(img, zoom = 0.3)
    annotation_box = AnnotationBbox(box, (0.15,0.2), frameon = False, xycoords='axes fraction')
    ax.add_artist(annotation_box)

fig.savefig('figures/figure_7.pdf', transparent=False, format="pdf", bbox_inches="tight")