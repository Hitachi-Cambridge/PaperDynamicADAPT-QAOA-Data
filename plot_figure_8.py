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

fig_noise, ax_noise = plt.subplots(figsize=(12,8))

for dir in os.listdir(output_dir):
    
    if 'DS_Store' in dir:
        continue

    if dir == 'dynamic_adapt_noisy':
        label = 'Dynamic-ADAPT-QAOA (noisy growth)'
        color='black'
        marker = 'o'
    elif dir == 'adapt_noisy':
        label = 'ADAPT-QAOA (noisy growth)'
        color = 'gray'
        marker = 's'
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
    ax_noise.errorbar(layers, 1-mean_approximation_ratios, std_approximation_ratios, label=label, color=color, marker=marker, alpha=0.6)
    

for dir in os.listdir(output_dir):
    
    if 'DS_Store' in dir:
        continue

    if dir == 'dynamic_adapt':
        label = 'Dynamic ADAPT-QAOA (noiseless growth)'
        color = '#fe6000'
    elif dir == 'adapt':
        label = 'ADAPT-QAOA (noiseless growth)'
        color = '#64a5ff'
    else:
        continue
    
    data_dir = output_dir + dir + '/' + 'noisy_data/'

    for file in os.listdir(data_dir):
        if 'DS_Store' in file:
            continue
        gate_error = float(file.split('_')[1])
        if gate_error != 0.00122:
            continue
        noisy_data = pd.read_csv(data_dir + file)
        noisy_data = noisy_data.to_numpy().transpose()[1:]
        
        mean_approximation_ratios = np.average(noisy_data, axis=0)
        std_approximation_ratios = 1.96 * np.std(noisy_data, axis=0) / np.sqrt(len(noisy_data))
        layers = [i for i in range(len(mean_approximation_ratios))]
        ax_noise.plot(layers, 1-mean_approximation_ratios, label=label, color=color)
        ax_noise.fill_between(layers, 1-(mean_approximation_ratios-std_approximation_ratios), 1-(mean_approximation_ratios+std_approximation_ratios), alpha=0.3,color=color)
        

ax_noise.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_noise.tick_params(axis='both', which='major', labelsize=fontsize_axis-2)
ax_noise.legend()
ax_noise.set_yscale('log')
# ax_noise.set_title('Data for Gate-Error Probability ' + str(gate_error))
ax_noise.set_ylabel(r'$1-\alpha$', fontsize=fontsize_axis)
ax_noise.set_xlabel('Circuit Depth, $P$', fontsize=fontsize_axis)
ax_noise.legend(loc='upper right', fontsize=15)

for index, no_nodes in enumerate([6]):

    img = plt.imread('./figures/' + str(no_nodes) + '_nodes.png')
    #The OffsetBox is a simple container artist.
    #The child artists are meant to be drawn at a relative position to its #parent.
    box = OffsetImage(img, zoom = 0.3)
    #Annotation box for solar pv logo
    #Container for the imagebox referring to a specific position *xy*.
    annotation_box = AnnotationBbox(box, (0.15,0.2), frameon = False, xycoords='axes fraction')
    ax_noise.add_artist(annotation_box)

fig_noise.savefig('figures/figure_8.pdf', transparent=False, format="pdf", bbox_inches="tight")