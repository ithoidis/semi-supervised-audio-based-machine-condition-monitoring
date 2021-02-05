import matplotlib.style
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#plt.style.use('seaborn-deep')
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

# plt.style.use('seaborn-colorblind')

#plt.subplots(3,3)
fig, [ax1, ax2] = plt.subplots(1,2, figsize = (6, 3),)
cm = plt.get_cmap('tab20b')
s = ['a', 'b', 'c']
figs = ['./fig1/vis_%d_ce.npy' % 40, './fig1/vis_%d_center.npy' % 56, './fig1/vis_%d_similarity.npy' % 37]
[feat, labels, classes, epoch] = np.load(figs[0], allow_pickle=True)

ax = plt.subplot(1,2,1)
plt.text(0.45,0.065, 'Activation of the 1st neuron', transform=plt.gcf().transFigure, ha='center', va='center')

ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
evenly_spaced_interval = np.linspace(0, 1, len(classes))
for i in range(len(classes)):
    plt.scatter(feat[labels == i, 0], feat[labels == i, 1],s=0.5)
plt.ylabel('Activation of the 2nd neuron')
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=True,  # ticks along the bottom edge are off
    top=True,  # ticks along the top edge are off
    labelbottom=False, direction='in')  # labels along the bottom edge are off
plt.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    left=True,  # ticks along the bottom edge are off
    right=True,  # ticks along the top edge are off
    labelleft=False, direction='in')  # labels along the bottom edge are off
plt.text(-61*2, 71*2, "$a$",fontsize=16, horizontalalignment='right',  verticalalignment='top')

plt.xlim(-150, 150)
plt.ylim(-150, 150)
#xlim, ylim = 3, 3


[feat, labels, classes, epoch] = np.load(figs[1], allow_pickle=True)
ax = plt.subplot(1,2,2)
ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
evenly_spaced_interval = np.linspace(0, 1, len(classes))
for i in range(len(classes)):
    plt.scatter(feat[labels == i, 0], feat[labels == i, 1],s=0.5)

plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=True,  # ticks along the bottom edge are off
    top=True,  # ticks along the top edge are off
    labelbottom=False, direction='in')  # labels along the bottom edge are off
plt.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    left=True,  # ticks along the bottom edge are off
    right=True,  # ticks along the top edge are off
    labelleft=False, direction='in')  # labels along the bottom edge are off

plt.text(-2.8, 3.5, "$b$",fontsize=16, horizontalalignment='right',  verticalalignment='top')
if False:
    [feat, labels, classes, epoch] = np.load(figs[2], allow_pickle=True)
    ax = plt.subplot(1,3,3)
    ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
    evenly_spaced_interval = np.linspace(0, 1, len(classes))
    for i in range(len(classes)):
        plt.scatter(feat[labels == i, 0], feat[labels == i, 1],s=0.5)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=False, direction='in')  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=True,  # ticks along the bottom edge are off
        right=True,  # ticks along the top edge are off
        labelleft=False, direction='in')  # labels along the bottom edge are off

    plt.text(-1.4, 1.7, "$c$",fontsize=16, horizontalalignment='right',  verticalalignment='top')
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.85)

#plt.xlabel('Activation of the 1st neuron')
legend = plt.legend(classes, title='Classes', bbox_to_anchor=(1.05, 1.05),  shadow = False, loc='upper left',fontsize='x-small', scatterpoints=5,)

# plt.tight_layout()
for handle in legend.legendHandles:
    handle.set_sizes([6.0])
plt.savefig('figure1.pdf', dpi=1000)

# plt.savefig('./images/epoch%d.pdf' % epoch)


