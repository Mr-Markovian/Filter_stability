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
mpl.rcParams['legend.fontsize']=30
mpl.rcParams['xtick.labelsize']=30
mpl.rcParams['ytick.labelsize']=30
mpl.rcParams['axes.labelsize']=30
mpl.rcParams['axes.titlesize']=40
mpl.rcParams['figure.figsize']=(10,10)

# def plot_for_me(data_path,plot_style,store_path,inset=False):
#     if inset=True:
#         ax.
#         plot_for_me()

# #Load data:
os.chdir('/home/shashank/Lorenz_63/10_dim_L96/codes/N=50')

#Different I.C.s for the filters N=50
dist16=np.load('a_distance_between_bias,cov=0.0_0.1and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')
dist17=np.load('a_distance_between_bias,cov=2.0_0.5and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')
dist18=np.load('a_distance_between_bias,cov=0.0_0.1and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')

dist16_a=np.load('a_distance_between_bias,cov=0.0_0.1and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')
dist17_a=np.load('a_distance_between_bias,cov=2.0_0.5and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')
dist18_a=np.load('a_distance_between_bias,cov=0.0_0.1and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')

os.chdir('/home/shashank/Lorenz_63/Paper/images')
x=np.arange(400)
x_o_50=np.repeat(x[:50],10)
x_o_400 = np.repeat(x[::4], 10)

fig, ax =plt.subplots()
sns.lineplot(x_o_400,dist16.flatten('C'),label='$i=1,j=3$')
sns.lineplot(x_o_400,dist17.flatten('C'),label='$i=2,j=3$')
sns.lineplot(x_o_400,dist18.flatten('C'),label='$i=1,j=2$')
ax.set_xlabel('assimilation step (n)')
ax.set_ylabel(r'$D_{\epsilon}$')
ax.set_title(r'L96,  $D_{\epsilon}(\pi^{E}_n(\mu_i),\pi^{E}_n(\mu_j))$')

# For inset figure:
ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])#)
ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
sns.lineplot(x_o_50,dist16_a[:50,:].flatten('C'),ax=ax_inset)
sns.lineplot(x_o_50,dist17_a[:50,:].flatten('C'),ax=ax_inset)
sns.lineplot(x_o_50,dist18_a[:50,:].flatten('C'),ax=ax_inset)
ax_inset.set_xlabel('')
ax_inset.set_ylabel('')
ax_inset.tick_params(axis='both', labelsize=20)

plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('stable_50_L96_{}dim.png'.format(10))
plt.show()

#------------------------N=200 plots--------------------------------------#
# N=200
# #Load data:
# os.chdir('/home/shashank/Lorenz_63/10_dim_L96/codes/N=200')
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
# sns.lineplot(x_o_400,dist4.flatten('C'),label='$i=1,j=3$')
# sns.lineplot(x_o_400,dist5.flatten('C'),label='$i=2,j=3$')
# sns.lineplot(x_o_400,dist6.flatten('C'),label='$i=1,j=2$')
# ax.set_xlabel(r'assimilation step$\ $(n)')
# ax.set_ylabel(r'$D_{\epsilon}$')
# ax.set_title(r'L96,  $D_{\epsilon}(\pi^{E}_n(\mu_i),\pi^{E}_n(\mu_j))$')

# # For inset figure:
# ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])#)
# ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
# sns.lineplot(x_o_50,dist4_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist5_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist6_a[:50,:].flatten('C'),ax=ax_inset)
# ax_inset.set_xlabel('')
# ax_inset.set_ylabel('')
# ax_inset.tick_params(axis='both', labelsize=20)

# plt.legend(frameon=True)
# plt.tight_layout()
# plt.savefig('stable_{}_L96_{}dim.png'.format(N,10))
# plt.show()
#----------------------------------------------------------------------#

# #Comparison between N=50 and N=200:
# os.chdir('/home/shashank/Lorenz_63/10_dim_L96/codes/N=50_vs_N=200')
# dist13=np.load('a_distance_between_bias,cov=4.0_1.0and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to400.npy')
# dist14=np.load('a_distance_between_bias,cov=2.0_0.5and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to400.npy')
# dist15=np.load('a_distance_between_bias,cov=0.0_0.1and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to400.npy')

# dist13_a=np.load('a_distance_between_bias,cov=4.0_1.0and4.0_1.0_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to100.npy')
# dist14_a=np.load('a_distance_between_bias,cov=2.0_0.5and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to100.npy')
# dist15_a=np.load('a_distance_between_bias,cov=0.0_0.1and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N1=50,N2=200_1to10_t=0to100.npy')

# os.chdir('/home/shashank/Lorenz_63/Paper/images')

# x=np.arange(400)
# x_o_50=np.repeat(x[:50],10)
# x_o_400 = np.repeat(x[::4], 10)

# fig, ax =plt.subplots()
# sns.lineplot(x_o_400,dist13.flatten('C'),label='$i=j=3$')
# sns.lineplot(x_o_400,dist14.flatten('C'),label='$i=j=2$')
# sns.lineplot(x_o_400,dist15.flatten('C'),label='$i=j=1$')
# ax.set_xlabel(r'assimilation step (n)')
# ax.set_ylabel(r'$D_{\epsilon}$')
# ax.set_title(r'$D_{\epsilon}(\pi^{E,50}_n(\mu_i),\pi^{E,200}_n(\mu_i))$')

# # For inset figure:
# ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])#)
# ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
# sns.lineplot(x_o_50,dist13_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist14_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist15_a[:50,:].flatten('C'),ax=ax_inset)
# ax_inset.set_xlabel('')
# ax_inset.set_ylabel('')
# ax_inset.tick_params(axis='both', labelsize=20)

# plt.legend(frameon=True)
# plt.tight_layout()
# plt.savefig('stable_50_vs_200_L96_{}dim.png'.format(10))
# plt.show()

# L96 40 dim -----------------------------------------------------------------

#Load data:
# os.chdir('/home/shashank/Lorenz_63/40_dim_L96/codes')

# #Different I.C.s for the filters N=50
# dist16=np.load('a_distance_between_bias,cov=4.0_1.0and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')
# dist17=np.load('a_distance_between_bias,cov=2.0_0.5and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')
# dist18=np.load('a_distance_between_bias,cov=4.0_1.0and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to400.npy')

# dist16_a=np.load('a_distance_between_bias,cov=4.0_1.0and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')
# dist17_a=np.load('a_distance_between_bias,cov=2.0_0.5and0.0_0.1_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')
# dist18_a=np.load('a_distance_between_bias,cov=4.0_1.0and2.0_0.5_for_mu=1.0,ob_gap=0.1_for_N=50_1to10_t=0to100.npy')

# os.chdir('/home/shashank/Lorenz_63/Paper/images')
# x=np.arange(400)
# x_o_50=np.repeat(x[:50],10)
# x_o_400 = np.repeat(x[::4], 10)

# fig, ax =plt.subplots()
# sns.lineplot(x_o_400,dist16.flatten('C'),label='$i=1,j=3$')
# sns.lineplot(x_o_400,dist17.flatten('C'),label='$i=2,j=3$')
# sns.lineplot(x_o_400,dist18.flatten('C'),label='$i=1,j=2$')
# ax.set_xlabel('assimilation step (n)')
# ax.set_ylabel(r'$D_{\epsilon}$')
# ax.set_title(r'L96,  $D_{\epsilon}(\pi^{E}_n(\mu_i),\pi^{E}_n(\mu_j))$')

# # For inset figure:
# ax_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])#)
# ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
# sns.lineplot(x_o_50,dist16_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist17_a[:50,:].flatten('C'),ax=ax_inset)
# sns.lineplot(x_o_50,dist18_a[:50,:].flatten('C'),ax=ax_inset)
# ax_inset.set_xlabel('')
# ax_inset.set_ylabel('')
# ax_inset.tick_params(axis='both', labelsize=20)

# plt.legend(frameon=True)
# plt.tight_layout()
# plt.savefig('stable_50_loc_L96_{}dim.png'.format(40))
# plt.show()

# ---------------------------sinkhorn distance------------------------------
# from matplotlib.cm import get_cmap
# cmname = 'tab20'
# cmp = get_cmap(cmname)
# cov=0.1
# os.chdir('/home/shashank/Lorenz_63/Codes//Wasserstein/notebooks')
# dat4=np.load('distances_sizes=50_to_1600_cov={}_realizations=20_dim=5_to_45_div_tf.npy'.format(cov))
# dat3=np.load('distances_sizes=50_to_1600_realizations=50_dim=5_to_45_div_tf.npy')[:,:,:20]

# dat1=np.load('distances_sizes=50_to_1600_cov=1.0_realizations=20_dim=3_to_3.npy')
# dat2=np.load('distances_sizes=50_to_1600_cov=0.1_realizations=20_dim=3_to_3.npy')

# #New numpy array to contain all:
# dat1=np.concatenate((dat1,dat3),axis=1)
# dat2=np.concatenate((dat2,dat4),axis=1)
# ens=np.array([50,100,200,400,800,1600])
# dims=np.concatenate((np.array([3]),np.arange(5,50,5)))

# np.save('distances_sizes=50_to_1600_cov=1.0_realizations=20_dim=3_to_45.npy',dat1)
# np.save('distances_sizes=50_to_1600_cov=0.1_realizations=20_dim=3_to_45.npy',dat2)
# np.save('dims_3_to_45.npy',dims)
# os.chdir('/home/shashank/Lorenz_63/Paper/images')

# for i in range(6):
#     plt.plot(dims,np.mean(dat2[i],axis=1),marker='o',c=cmp(i/6),linestyle='dashdot')
#     plt.plot(dims,np.mean(dat1[i],axis=1),marker='o',label='m={}'.format(ens[i]),c=cmp(i/6))
#     #plt.errorbar(dims,np.mean(dat4[i],axis=1), yerr=np.std(dat4[i],axis=1)/np.sqrt(50), fmt='o',alpha=1,markersize=5,
#     #                ecolor='saddlebrown', elinewidth=2, capsize=5,label='{}'.format(ens[i]))

# plt.text(25,6.5,r'$\lambda=1.0$',fontsize=30)
# plt.text(25,2.2,r'$\lambda=0.1$',fontsize=30)
# plt.xlabel(r'dimension')
# plt.xticks(dims)
# plt.ylabel(r'$D_{\epsilon}$')
# plt.title(r'$D_{\epsilon}$ for two samples $\sim\mathcal{N}(0,\lambda I)$')
# plt.legend(frameon='true',fontsize=25)
# plt.tight_layout()
# plt.savefig('zeros_cov={}_cov=1.0.png'.format(cov))
# plt.show()