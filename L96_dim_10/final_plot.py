import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib as mpl
import seaborn as sns

# Codes for generating final 
plt.style.use('seaborn-paper')

mpl.rcParams['lines.markersize']=10
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=20
mpl.rcParams['ytick.labelsize']=20
mpl.rcParams['axes.labelsize']=20
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['figure.figsize']=(10,10)

def plot_for_me(data_path,plot_style,store_path,inset=False):
    if inset=True:
        ax.
        plot_for_me()

#Load data:
# os.chdir('/home/shashank/Lorenz_63/10_dim_L96/codes/N=50')

# #Different I.C.s for the filters N=50
# dist16=np.load('a_distance_between_bias,cov=0.0_0.1and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')
# dist17=np.load('a_distance_between_bias,cov=2.0_0.5and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')
# dist18=np.load('a_distance_between_bias,cov=0.0_0.1and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')

# dist16_a=np.load('a_distance_between_bias,cov=0.0_0.1and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')
# dist17_a=np.load('a_distance_between_bias,cov=2.0_0.5and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')
# dist18_a=np.load('a_distance_between_bias,cov=0.0_0.1and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')

# os.chdir('/home/shashank/Lorenz_63/Paper/images')
# x=np.arange(400)
# x_o_50=np.repeat(x[:50],10)
# x_o_400 = np.repeat(x[::4], 10)

# fig, ax =plt.subplots()
# sns.lineplot(x_o_400,dist16.flatten('C'),label='$D(\pi^{E}_n(\mu_1),\pi^{E}_n(\mu_3))$')
# sns.lineplot(x_o_400,dist17.flatten('C'),label='$D(\pi^{E}_n(\mu_2),\pi^{E}_n(\mu_3))$',color='g')
# sns.lineplot(x_o_400,dist18.flatten('C'),label='$D(\pi^{E}_n(\mu_1),\pi^{E}_n(\mu_2))$',color='r')
# ax.set_xlabel(r'assimilation step$(n)$')
# ax.set_ylabel(r'$D_{\epsilon}$')
# ax.set_title(r'Stability for N=50,ocov=1.0')

# # For inset figure:
# ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])#)
# ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
# sns.lineplot(x_o_50,dist16_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist17_a[:50,:].flatten('C'),ax=ax_inset,color='g')
# sns.lineplot(x_o_50,dist18_a[:50,:].flatten('C'),ax=ax_inset,color='r')
# ax_inset.set_xlabel('')
# ax_inset.set_ylabel('')
# ax_inset.tick_params(axis='both', labelsize=14)

# plt.legend(frameon=True)
# plt.savefig('stable_50_L96_{}dim.jpg'.format(10))
# plt.show()

#------------------------N=200 plots--------------------------------------#
#Load data:
#os.chdir('/home/shashank/Lorenz_63/10_dim_L96/codes/N=200')
# # Different I.C.s for N=200
# dist4=np.load('a_distance_between_bias,cov=4.0_1.0and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=200_1to10_t=0to400.npy')
# dist5=np.load('a_distance_between_bias,cov=4.0_1.0and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=200_1to10_t=0to400.npy')
# dist6=np.load('a_distance_between_bias,cov=2.0_0.5and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=200_1to10_t=0to400.npy')

# dist4_a=np.load('a_distance_between_bias,cov=4.0_1.0and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=200_1to10_t=0to100.npy')
# dist5_a=np.load('a_distance_between_bias,cov=4.0_1.0and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=200_1to10_t=0to100.npy')
# dist6_a=np.load('a_distance_between_bias,cov=2.0_0.5and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=200_1to10_t=0to100.npy')

# os.chdir('/home/shashank/Lorenz_63/Paper/images')
# x=np.arange(400)
# x_o_50=np.repeat(x[:50],10)
# x_o_400 = np.repeat(x[::4], 10)
# fig, ax =plt.subplots()
# sns.lineplot(x_o_400,dist4.flatten('C'),label='$D(\pi^{E}_n(\mu_1),\pi^{E}_n(\mu_3))$')
# sns.lineplot(x_o_400,dist5.flatten('C'),label='$D(\pi^{E}_n(\mu_2),\pi^{E}_n(\mu_3))$',color='g')
# sns.lineplot(x_o_400,dist6.flatten('C'),label='$D(\pi^{E}_n(\mu_1),\pi^{E}_n(\mu_2))$',color='r')
# ax.set_xlabel(r'assimilation step$(n)$')
# ax.set_ylabel(r'$D_{\epsilon}$')
# ax.set_title(r'Stability for N=50,ocov=1.0')

# # For inset figure:
# ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])#)
# ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
# sns.lineplot(x_o_50,dist4_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist5_a[:50,:].flatten('C'),ax=ax_inset,color='g')
# sns.lineplot(x_o_50,dist6_a[:50,:].flatten('C'),ax=ax_inset,color='r')
# ax_inset.set_xlabel('')
# ax_inset.set_ylabel('')
# ax_inset.tick_params(axis='both', labelsize=14)

# plt.legend(frameon=True)
# plt.savefig('stable_200_L96_{}dim.jpg'.format(10))
# plt.show()

#Comparison between N=50 and N=200:
os.chdir('/home/shashank/Lorenz_63/10_dim_L96/codes/N=50_vs_N=200')
dist13=np.load('a_distance_between_bias,cov=4.0_1.0and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to400.npy')
dist14=np.load('a_distance_between_bias,cov=2.0_0.5and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to400.npy')
dist15=np.load('a_distance_between_bias,cov=0.0_0.1and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to400.npy')

dist13_a=np.load('a_distance_between_bias,cov=4.0_1.0and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to100.npy')
dist14_a=np.load('a_distance_between_bias,cov=2.0_0.5and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to100.npy')
dist15_a=np.load('a_distance_between_bias,cov=0.0_0.1and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to100.npy')

os.chdir('/home/shashank/Lorenz_63/Paper/images')

x=np.arange(400)
x_o_50=np.repeat(x[:50],10)
x_o_400 = np.repeat(x[::4], 10)

fig, ax =plt.subplots()
sns.lineplot(x_o_400,dist13.flatten('C'),label='$D(\pi^{E,50}_n(\mu_3),\pi^{E,200}_n(\mu_3))$')
sns.lineplot(x_o_400,dist14.flatten('C'),label='$D(\pi^{E,50}_n(\mu_2),\pi^{E,200}_n(\mu_2))$',color='g')
sns.lineplot(x_o_400,dist15.flatten('C'),label='$D(\pi^{E,50}_n(\mu_1),\pi^{E,300}_n(\mu_1))$',color='r')
ax.set_xlabel(r'assimilation step$(n)$')
ax.set_ylabel(r'$D_{\epsilon}$')
ax.set_title(r'Stability for N=50 versus N=200,ocov=1.0')

# For inset figure:
ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])#)
ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
sns.lineplot(x_o_50,dist13_a[:50,:].flatten('C'),ax=ax_inset)
sns.lineplot(x_o_50,dist14_a[:50,:].flatten('C'),ax=ax_inset,color='g')
sns.lineplot(x_o_50,dist15_a[:50,:].flatten('C'),ax=ax_inset,color='r')
ax_inset.set_xlabel('')
ax_inset.set_ylabel('')
ax_inset.tick_params(axis='both', labelsize=14)

plt.legend(frameon=True)
plt.savefig('stable_50_vs_200_L96_{}dim.jpg'.format(10))
plt.show()


# for N=200
# x=np.arange(400)[::4]
# x_o_400 = np.repeat(x, 9)
# y4_for_lineplot = dist4[:,[0,1,3,4,5,6,7,8,9]].flatten('C')
# y5_for_lineplot = dist5[:,[0,1,3,4,5,6,7,8,9]].flatten('C')
# y6_for_lineplot = dist6[:,[0,1,3,4,5,6,7,8,9]].flatten('C')
# sns.lineplot(x_o_400,y4_for_lineplot,label='$d(\mu_1,\mu_3)$')
# sns.lineplot(x_o_400,y5_for_lineplot,label='$d(\mu_2,\mu_3)$',color='g')
# sns.lineplot(x_o_400,y6_for_lineplot,label='$d(\mu_1,\mu_2)$',color='r')
# plt.xlabel(r'assimilation step')
# plt.ylabel(r'$\sqrt{S_{\epsilon} }$')
# plt.title('Stability for N=200, ocov=1.0')
# plt.legend(frameon=True)
# plt.savefig('stable_200.jpg')
# plt.show()

# os.chdir('/home/shashank/Lorenz_63/Codes//Wasserstein/notebooks')
# dat4=np.load('distances_sizes=50_to_1600_realizations=50_dim=5_to_45_div_tf.npy')
# ens=np.array([50,100,200,400,800,1600])
# dims=np.arange(5,50,5)

# os.chdir('/home/shashank/Lorenz_63/Paper/images')

# for i in range(6):
#     plt.plot(dims,np.mean(dat4[i],axis=1),marker='o',label='{}'.format(ens[i]))
#     #plt.errorbar(dims,np.mean(dat4[i],axis=1), yerr=np.std(dat4[i],axis=1)/np.sqrt(50), fmt='o',alpha=1,markersize=5,
#     #                ecolor='saddlebrown', elinewidth=2, capsize=5,label='{}'.format(ens[i]))
# plt.xlabel(r'dimension')
# plt.ylabel(r'$D_{\epsilon}$')
# plt.title('Distance between two ensemble $\sim\mathcal{N}(0,I)$')
# plt.legend(frameon='true')
# plt.savefig('zeros.jpg')
# plt.show()