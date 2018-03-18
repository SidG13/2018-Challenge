import json, pandas as pd, numpy as np
import scipy
import csv,matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = json.load(open('data.json'))

list_of_categories = sorted(list(data.keys()))
list_of_genes = list(data.values())
flat_list = [item for sublist in list_of_genes for item in sublist]
set_of_genes = sorted(set(flat_list))

print(set_of_genes)
# we have all the genes, and we have all the traits
# create an empyt matrix from this
# ok, best is: just to do it manually
# matrix = np.column_stack((list_of_categories,list_of_genes) )

mapping_keys = {}
for index, key in enumerate(set_of_genes):
    mapping_keys[key]  = index

print(mapping_keys)
matrix = []


for trait,genes in sorted(data.items()):
    print (trait)
    # look it up in the mapping
    row = [0] * len(set_of_genes)
    for gene in genes:
        row[mapping_keys[gene]] = 1
    matrix.append(row)


#
print(matrix)
np_mat = np.matrix(matrix)
print(np_mat[0])
print(np_mat.shape)
trans_mat  = np_mat.transpose()
print(trans_mat.shape)

# df = pd.DataFrame(data)

#dataframe: add headers and such

df = pd.DataFrame(trans_mat)
df.columns=list_of_categories
x = df.loc[: , list_of_categories].values
x = StandardScaler().fit_transform(x)
print(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1'])
print(principalDf)

pca_dict = {}
print("hello")
for index, gene in enumerate(set_of_genes):
    # print("this is the " + gene)
    pca_dict[gene] = principalDf.iloc[index]["principal component 1"]
print(pca_dict  )
plt.plot(pca_dict.values())
plt.show()

for val  in pca_dict.values():
    if val > 6: print("print good")
# with open('pca.csv', 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile,fieldnames=["gene", "pca"])



'''
====================
Customized colorbars
====================

This example shows how to build colorbars without an attached mappable.
'''

import matplotlib.pyplot as plt
import matplotlib as mpl

# Make a figure and axes with dimensions as desired.
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=5, vmax=10)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb1.set_label('Some Units')

# The second example illustrates the use of a ListedColormap, a
# BoundaryNorm, and extended ends to show the "over" and "under"
# value colors.
cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = sorted(list(pca_dict.values()))
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                norm=norm,
                                # to use 'extend', you must
                                # specify two extra boundaries:
                                boundaries=[0] + bounds + [13],
                                extend='both',
                                ticks=bounds,  # optional
                                spacing='proportional',
                                orientation='horizontal')
cb2.set_label('Discrete intervals, some other units')

# The third example illustrates the use of custom length colorbar
# extensions, used on a colorbar with discrete intervals.
cmap = mpl.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],
                                  [1., .8, 0.], [1., .4, 0.]])
cmap.set_over((1., 0., 0.))
cmap.set_under((0., 0., 1.))

bounds = [-1., -.5, 0., .5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                norm=norm,
                                boundaries=[-10] + bounds + [10],
                                extend='both',
                                # Make the length of each extension
                                # the same as the length of the
                                # interior colors:
                                extendfrac='auto',
                                ticks=bounds,
                                spacing='uniform',
                                orientation='horizontal')
cb3.set_label('Custom extension lengths, some other units')

plt.show()

# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# y-axis in bold
rc('font', weight='bold')

# Values of each group
bars1 = [12, 28, 1, 8, 22]
bars2 = [28, 7, 16, 4, 10]
bars3 = [25, 3, 23, 25, 17]

# Heights of bars1 + bars2 (TO DO better)
bars = [40, 35, 17, 12, 32]

# The position of the bars on the x-axis
r = [0, 1, 2, 3, 4]

# Names of group and bar width
names = ['A', 'B', 'C', 'D', 'E']
barWidth = 1

# Create brown bars
plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
# Create green bars (middle), on top of the firs ones
plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
# Create green bars (top)
plt.bar(r, bars3, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)

# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")

# Show graphic
plt.show()
