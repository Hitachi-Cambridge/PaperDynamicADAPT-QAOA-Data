import os, sys
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

layer = 12

fig, ax = plt.subplots(figsize=(12,8))

for dir in os.listdir(output_dir):
    
    if 'DS_Store' in dir:
        continue

    if dir == 'dynamic_adapt':
        label = 'Original'
        marker = 'o'
        color = '#fe6000'
        linewidth=2
    elif dir == 'adapt':
        continue
        label = 'ADAPT-QAOA (noisless growth)'
        marker = 's'
        color = '#64a5ff'
    else:
        continue
    
    data_dir = output_dir + dir + '/' + 'noisy_data/'

    mean_approximation_ratios = []
    std_approximation_ratios = []
    gate_error_probs = []
    file_strings = []

    for file in os.listdir(data_dir):
        if 'DS_Store' in file:
            continue
        gate_error = float(file.split('_')[1])
        file_strings.append(file)
        # print(gate_error)
        gate_error_probs.append(gate_error)
        noisy_data = pd.read_csv(data_dir + file)
        noisy_data = noisy_data.to_numpy().transpose()[1:]
        best_approx_ratios = np.max(noisy_data, axis=1)
        approx_ratios = noisy_data[:,layer]
        
        mean_approximation_ratio = np.average(approx_ratios)
        std_approximation_ratio = np.std(approx_ratios) / np.sqrt(len(approx_ratios))
        mean_approximation_ratios.append(mean_approximation_ratio)
        std_approximation_ratios.append(std_approximation_ratio)

    mean_approximation_ratios=np.array(mean_approximation_ratios)
    std_approximation_ratios=np.array(std_approximation_ratios)
    gate_error_probs = np.array(gate_error_probs)
    file_strings = np.array(file_strings)
    arg_sort = np.argsort(gate_error_probs)
    gate_error_probs = gate_error_probs[arg_sort]
    mean_approximation_ratios = mean_approximation_ratios[arg_sort]
    std_approximation_ratios = std_approximation_ratios[arg_sort]
    file_strings = file_strings[arg_sort]
    
    ax.errorbar(gate_error_probs, mean_approximation_ratios, std_approximation_ratios, label=label, color=color, linewidth=linewidth, capsize=5,alpha=0.8)
    

# perform error mitigation

for dir in os.listdir(output_dir):
    
    if 'DS_Store' in dir:
        continue

    if dir == 'dynamic_adapt':
        label = 'Error-Mitigated'
        marker = 'o'
        color = '#77cca4'
        linewidth=2
    elif dir == 'adapt':
        continue
        label = 'ADAPT-QAOA (noisless growth)'
        marker = 's'
        color = '#64a5ff'
    else:
        continue
    
    data_dir = output_dir + dir + '/' + 'noisy_data/'

    new_mean_approximation_ratios = []
    new_std_approximation_ratios = []

    for index, gate_error in enumerate(gate_error_probs[:-1]):

        noise_prob_1 = gate_error
        noise_prob_2 = gate_error_probs[index + 1]

        c = noise_prob_2 / noise_prob_1
        gamma_1 = c/(c-1)
        gamma_2 = 1/(1-c)

        noisy_data_1 = pd.read_csv(data_dir + file_strings[index])
        noisy_data_1 = noisy_data_1.to_numpy().transpose()[1:]
        noisy_data_2 = pd.read_csv(data_dir + file_strings[index+1])
        noisy_data_2 = noisy_data_2.to_numpy().transpose()[1:]

        new_noisy_data = gamma_1 * noisy_data_1 + gamma_2 * noisy_data_2

        best_approx_ratios = np.max(new_noisy_data, axis=1)
        approx_ratios = new_noisy_data[:,layer]
        
        mean_approximation_ratio = np.average(approx_ratios)
        std_approximation_ratio = np.std(approx_ratios) / np.sqrt(len(approx_ratios))
        new_mean_approximation_ratios.append(mean_approximation_ratio)
        new_std_approximation_ratios.append(std_approximation_ratio)

    ax.errorbar(gate_error_probs[:-1], new_mean_approximation_ratios, new_std_approximation_ratios, label=label, color=color, linewidth=linewidth, capsize=5, alpha=0.8)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis='both', which='major', labelsize=fontsize_axis-2)
ax.legend()
ax.set_xscale('log')
ax.set_ylabel(r'$\alpha$', fontsize=fontsize_axis)
ax.set_xlabel('Gate-error probability, $p_\mathrm{gate}$', fontsize=fontsize_axis)
ax.legend(loc='upper right', fontsize=15)

for index, no_nodes in enumerate([6]):

    img = plt.imread('./figures/' + str(no_nodes) + '_nodes.png')
    box = OffsetImage(img, zoom = 0.3)
    annotation_box = AnnotationBbox(box, (0.15,0.2), frameon = False, xycoords='axes fraction')
    ax.add_artist(annotation_box)

fig.savefig('figures/figure_9.pdf', transparent=False, format="pdf", bbox_inches="tight")