import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

data_dir = './output_data/'
colors = ['#64a5ff', '#fe6000']
labels = ['ADAPT-QAOA', 'Dynamic-ADAPT-QAOA']
fontsize_axis = 23

fig_hist, ax_hist = plt.subplots(figsize=(12,4))

all_ham_parameters = {}
max_normalisation_ham = 0

img = plt.imread('./figures/6_nodes.png')
box = OffsetImage(img, zoom = 0.3)
annotation_box = AnnotationBbox(box, (0.15,0.65), frameon = False, xycoords='axes fraction')
ax_hist.add_artist(annotation_box)

for index, algo_type in enumerate(['adapt_qaoa','dynamic_adapt_qaoa']):

    color = colors[index]
    label = labels[index]
    add = ''
    if index == 1:
        add = '_best_delta_2'

    all_ham_parameters[algo_type] = []
    
    for file in os.listdir(data_dir + algo_type + '/6_nodes' + add + '/convergence_data/'):

        if '.DS_Store' in file:
            continue

        seed = file.split('.')[0].split('_')[-1]

        data = pd.read_csv(data_dir + algo_type + '/6_nodes' + add + '/convergence_data/' + file)
        ham_params = data['Best Hamiltonian Param'].dropna().to_list()

        for index, param in enumerate(ham_params):

            while ham_params[index] > np.pi:
                ham_params[index] -= 2*np.pi
            while ham_params[index] < -1.0 * np.pi:
                ham_params[index] += 2*np.pi

        all_ham_parameters[algo_type] += ham_params

    ax_hist.hist(all_ham_parameters[algo_type], 315, color=color, label=label, density=False)

ax_hist.set_ylabel('Parameter count', fontsize=fontsize_axis, fontname='Times')
ax_hist.set_xlabel('Parameter value, $\gamma^\star$', fontsize=fontsize_axis, fontname='Times')

ax_hist.legend(loc='upper right', fontsize=20)
ax_hist.set_yscale('log')
ax_hist.set_xticks(ticks=np.array([-1,-3/4,-1/2,-1/4,0,1/4,1/2,3/4,1])*np.pi, labels=['$-\pi$', '$-\dfrac{3\pi}{4}$', '$-\dfrac{\pi}{2}$', '$-\dfrac{\pi}{4}$', '$0$', '$\dfrac{\pi}{4}$', '$\dfrac{\pi}{2}$', '$\dfrac{3\pi}{4}$', '$\pi$'], fontsize=fontsize_axis-5)
ax_hist.tick_params(axis='both', which='major', labelsize=fontsize_axis-2)
fig_hist.savefig('figures/figure_3.pdf', transparent=False, format="pdf", bbox_inches="tight")
