import sys
sys.path.append('..')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

no_nodes = 6
max_noise_prob = 0.02
fontsize_axis = 23

fig_6, ax_6 = plt.subplots(figsize=(12,10))

ylim = 1.0

data_dir = 'output_data/noisy_data/' + str(no_nodes) + '_nodes/'

with open('output_data/goemans_williamson/' + str(no_nodes) + '_nodes/average_approximation_ratio.txt', 'r') as f:

    gw_data = f.readlines()[1]
    gw_average = float(gw_data.split(',')[0])
    gw_upper_ci = float(gw_data.split(',')[1].split('-')[0])
    gw_lower_ci = float(gw_data.split(',')[1].split('-')[1])
    x_pos = np.linspace(0.0, 1.0, 10)
    if no_nodes == 6:
        ax_6.plot(x_pos, [gw_average] * len(x_pos), label='Goemans-Williamson', linestyle='dashed', color='#77cca4', linewidth=2)
        ax_6.fill_between(x_pos, [gw_upper_ci] * len(x_pos), [gw_lower_ci] * len(x_pos), alpha=0.3, color='#77cca4')

for subfolder in os.listdir(data_dir + 'noisy_approximation_ratios'):

    if '.DS_Store' in subfolder:
        continue
    if 'dynamic' in subfolder:
        label = 'Dynamic ADAPT-QAOA'
        linestyle = 'solid'
        linewidth=2.5
        color = '#fe6202'
    elif 'adapt' in subfolder:
        label = 'ADAPT-QAOA'
        linestyle = 'dashdot'
        linewidth=1.5
        color = '#64a5ff'

    noise_probabilities = []
    best_approx_ratios = []
    best_approx_ratios_errors = []

    for noise_prob_data in os.listdir(data_dir + 'noisy_approximation_ratios/' + subfolder):

        if '.DS_Store' in noise_prob_data:
            continue

        noise_prob = float(noise_prob_data.split('_')[1])
        noise_probabilities.append(noise_prob)
        approx_ratio_data = pd.read_csv(data_dir + 'noisy_approximation_ratios/' + subfolder + '/' + noise_prob_data, header=0)
        approx_ratio_data.drop([approx_ratio_data.columns[0]], axis=1, inplace=True)

        max_approx_ratios_per_seed = approx_ratio_data.max(axis=0, skipna=True).to_list()
        max_approx_ratio_average = np.average(max_approx_ratios_per_seed)
        max_approx_ratio_error_in_average = np.std(max_approx_ratios_per_seed) / np.sqrt(len(max_approx_ratios_per_seed))

        best_approx_ratios.append(max_approx_ratio_average)
        best_approx_ratios_errors.append(max_approx_ratio_error_in_average)

    noise_probabilities_sorted_indices = np.argsort(noise_probabilities)
    best_approx_ratios = np.array(best_approx_ratios)[noise_probabilities_sorted_indices]
    best_approx_ratios_errors = np.array(best_approx_ratios_errors)[noise_probabilities_sorted_indices]
    noise_probabilities = np.array(noise_probabilities)[noise_probabilities_sorted_indices]

    max_index = len(noise_probabilities)
    for index, noise_prob in enumerate(noise_probabilities):
        if noise_prob > max_noise_prob:
            max_index = index
            break
    max_index += 1
    if min(best_approx_ratios[:max_index]) < ylim:
        ylim = min(best_approx_ratios[:max_index])

    ax_6.errorbar(noise_probabilities[:max_index], np.array(best_approx_ratios)[:max_index], best_approx_ratios_errors[:max_index], label=label, capsize=4, elinewidth=0.5, linestyle=linestyle, color=color, linewidth=linewidth)

ax_6.set_xlim(left = -0.0001, right=1.01 * max_noise_prob)
ax_6.set_xlabel('$p_\mathrm{gate}$', fontsize=fontsize_axis)
ax_6.set_ylabel(r'$\alpha^\star$', fontsize=fontsize_axis)
ax_6.tick_params(axis='both', which='major', labelsize=fontsize_axis-2)
ax_6.legend(loc='upper right', fontsize=fontsize_axis-4)
fig_6.text(0.83,0.045,r'$\times 10^{-2}$', fontsize=fontsize_axis-2)

for index, no_nodes in enumerate([6]):

    img = plt.imread('figures/' + str(no_nodes) + '_nodes.png')
    box = OffsetImage(img, zoom = 0.3)
    annotation_box = AnnotationBbox(box, (0.15,0.15), frameon = False, xycoords='axes fraction')
    ax_6.add_artist(annotation_box)

fig_6.savefig('figures/figure_5.pdf', transparent=False, format="pdf", bbox_inches="tight")