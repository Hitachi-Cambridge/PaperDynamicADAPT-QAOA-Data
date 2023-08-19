import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')
import sys
sys.path.append('..')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

fontsize_axis = 23
no_nodes = '6'
max_noise_prob = 0.004
num_noisy_graphs = 2

fig, ax_cnots = plt.subplots(1, 1, figsize = (12, 10))
max_num_layers = 0

data_dir = 'output_data/noisy_data/' + no_nodes + '_nodes/'

cnot_averages = {}
max_cnot_count = 0
counter=0

img = plt.imread('./figures/6_nodes.png')
box = OffsetImage(img, zoom = 0.3)
annotation_box = AnnotationBbox(box, (0.15,0.15), frameon = False, xycoords='axes fraction')
ax_cnots.add_artist(annotation_box)

for folder in os.listdir(data_dir):

    if 'cnot' not in folder:
        continue

    for subfolder in os.listdir(data_dir + folder):

        with open(data_dir + folder + '/' + subfolder + '/cnot_averages.txt', 'r') as f:

            data = f.readline().split(',')
            data_float = [float(entry) for entry in data]
            cnot_averages[subfolder] = data_float
            if max_cnot_count < data_float[-1]:
                max_cnot_count = data_float[-1]

with open('output_data/goemans_williamson/' + str(no_nodes) + '_nodes/average_approximation_ratio.txt', 'r') as f:

    gw_data = f.readlines()[1]
    gw_average = float(gw_data.split(',')[0])
    gw_upper_ci = float(gw_data.split(',')[1].split('-')[0])
    gw_lower_ci = float(gw_data.split(',')[1].split('-')[1])

    ax_cnots.plot([0.0, max_cnot_count], [1-gw_average] * 2, label='Goemans-Williamson', linestyle='dashed', color='#77cca4')
    ax_cnots.fill_between([0.0, max_cnot_count], [1-gw_upper_ci] * 2, [1-gw_lower_ci] * 2, alpha=0.3, color='#77cca4')


for subfolder in ['adapt_qaoa', 'dynamic_adapt_qaoa']:

    counter = 0

    if 'dynamic' in subfolder:
        label = 'Dynamic-ADAPT-QAOA'
        linestyles = ['solid', 'dashed', 'dashdot']
        colors=['#fe6000']*3
        opacity = [1.0, 0.75, 0.5]
        linewidth=2.5
    elif 'adapt' in subfolder:
        label = 'ADAPT-QAOA'
        linestyles = ['solid', 'dashed', 'dashdot']
        colors=['#64a5ff']*3
        opacity = [1.0, 0.75, 0.5]
        linewidth=1.5

    noise_probabilities = []

    for noise_prob_data in os.listdir(data_dir + 'noisy_approximation_ratios/' + subfolder):

        if '.DS_Store' in noise_prob_data:
            continue

        noise_prob = float(noise_prob_data.split('_')[1])
        noise_probabilities.append(noise_prob)

    noise_probabilities = sorted(noise_probabilities)

    cut_index = 0
    for index, noise_prob in enumerate(noise_probabilities):
        if noise_prob > max_noise_prob:
            cut_index = index
            break
    
    if cut_index > num_noisy_graphs + 1:
        tmp_num = (cut_index) / (num_noisy_graphs + 1)
        indices = [round(0 + tmp_num * i) for i in range(num_noisy_graphs+1)]
    else:
        indices = [i for i in range(num_noisy_graphs + 1)]
    noise_probabilities = np.array(noise_probabilities)[indices]

    for noise_prob_data in os.listdir(data_dir + 'noisy_approximation_ratios/' + subfolder):

        if '.DS_Store' in noise_prob_data:
            continue

        noise_prob = float(noise_prob_data.split('_')[1])

        if noise_prob not in noise_probabilities:
            continue

        approx_ratio_data = pd.read_csv(data_dir + 'noisy_approximation_ratios/' + subfolder + '/' + noise_prob_data, header=0)
        approx_ratio_data.drop([approx_ratio_data.columns[0]], axis=1, inplace=True)
        
        avg_approx_ratios = approx_ratio_data.mean(axis=1, skipna=True).to_list()
        conf_int_approx_ratios = 1.96 * np.array(approx_ratio_data.sem(axis=1, skipna=True).to_list())

        if max_num_layers < len(avg_approx_ratios):
            max_num_layers = len(avg_approx_ratios)

        if noise_prob == noise_probabilities[0]:
            counter=0
            ax_cnots.plot(cnot_averages[subfolder], 1-np.array(avg_approx_ratios), label = label, linestyle=linestyles[counter], color=colors[counter], alpha=opacity[counter], linewidth=linewidth)
            ax_cnots.fill_between(cnot_averages[subfolder], 1-np.array(avg_approx_ratios) - conf_int_approx_ratios, 1-np.array(avg_approx_ratios) + conf_int_approx_ratios, color=colors[counter], alpha=opacity[counter]*0.3)
        elif noise_prob == noise_probabilities[1]:
            counter = 1
            ax_cnots.plot(cnot_averages[subfolder], 1-np.array(avg_approx_ratios), linestyle=linestyles[counter], color=colors[counter], alpha=opacity[counter], linewidth=linewidth)
            ax_cnots.fill_between(cnot_averages[subfolder], 1-np.array(avg_approx_ratios) - conf_int_approx_ratios, 1-np.array(avg_approx_ratios) + conf_int_approx_ratios, color=colors[counter], alpha=opacity[counter]*0.3)
            ax_cnots.scatter(cnot_averages[subfolder][np.argmax(avg_approx_ratios)],1-np.max(avg_approx_ratios),marker='*',color='#0f0f0e',zorder=10,s=50)
        elif noise_prob == noise_probabilities[2]:
            counter = 2
            ax_cnots.plot(cnot_averages[subfolder], 1-np.array(avg_approx_ratios), linestyle=linestyles[counter], color=colors[counter], alpha=opacity[counter], linewidth=linewidth)
            ax_cnots.fill_between(cnot_averages[subfolder], 1-np.array(avg_approx_ratios) - conf_int_approx_ratios, 1-np.array(avg_approx_ratios) + conf_int_approx_ratios, color=colors[counter], alpha=opacity[counter]*0.3)
            ax_cnots.scatter(cnot_averages[subfolder][np.argmax(avg_approx_ratios)],1-np.max(avg_approx_ratios),marker='*',color='#0f0f0e',zorder=10,s=50)


ax_cnots.set_xlabel('Number of CNOT gates', fontsize=fontsize_axis)
ax_cnots.set_ylabel(r'$1-\alpha$', fontsize=fontsize_axis)
ax_cnots.set_yscale('log')
ax_cnots.set_ylim(0.0003, 0.5)
ax_cnots.legend(loc='upper right', fontsize=fontsize_axis-4)
ax_cnots.tick_params(axis='both', which='major', labelsize=fontsize_axis-2)

fig.savefig('figures/figure_4.pdf', transparent=False, format="pdf", bbox_inches="tight")