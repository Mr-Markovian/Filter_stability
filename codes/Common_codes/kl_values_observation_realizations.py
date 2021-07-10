import numpy as np
import os
from compare_dist import *
#os.chdir('/home/shashank/Lorenz_63/Trajectory_Observations')
os.chdir('/home/shashank/Lorenz_63/Comparison_PF_Enkf')
str_r=os.getcwd()
factor=1
mu=0.1
ob_gap=np.round(0.2*factor,2)

N=50 #,120,140
l_scale=0
alpha=1.0
ind_=3
bias=3.0
lambda_=1.0

#Go inside the data folder......................................
file_label='{}_bias={}_obs=2_ens=50_Mcov={},ocov=0.1_,gap=0.2_alpha=1.0_loc=convex_r=0'.format(ind_,bias,lambda_)
os.chdir(file_label)

#Load data....
f_ens_b=np.load(file_label+'f_ensemble.npy')
a_ens_b=np.load(file_label+'a_ensemble.npy') #ens has shape:=[time steps,system dimension,ensemble number]
f_mean_b=np.sum(f_ens_b,axis=2)/N
a_mean_b=np.sum(a_ens_b,axis=2)/N
#time=np.load(file_label+'time.npy')

# #Varying bias:
# kl_vals=np.zeros((50,2,3))
# _bias=np.array([1.0,2.0,3.0])
# # Getting..the access to other folders..
# for k,j in enumerate(_bias):
#     file_label='{}_bias={}_obs=2_ens=50_Mcov={},ocov=0.1_,gap=0.2_alpha=1.0_loc=convex_r=0'.format(ind_,j,lambda_)
#     os.chdir(str_r+'/'+file_label)
#     f_ens=np.load(file_label+'f_ensemble.npy')
#     a_ens=np.load(file_label+'a_ensemble.npy')
#     for i in range(50):
#         comp_f=DistComparison(f_ens_b[i].T,f_ens[i].T)
#         comp_a=DistComparison(a_ens_b[i].T,a_ens[i].T)
#         kl_vals[i,0,k]=comp_f.compute_KL_with_weights()
#         kl_vals[i,1,k]=comp_a.compute_KL_with_weights()
# os.chdir('/home/shashank/Lorenz_63/Comparison_PF_Enkf')
# np.save('kl_vals_fixed_lambda={}.npy'.format(lambda_),kl_vals)

#Varying variance:

kl_vals=np.zeros((50,2,3))
_var=np.array([1.0,0.1,0.01])
# Getting..the access to other folders..
for k,j in enumerate(_var):
    file_label='{}_bias={}_obs=2_ens=50_Mcov={},ocov=0.1_,gap=0.2_alpha=1.0_loc=convex_r=0'.format(ind_,bias,j)
    os.chdir(str_r+'/'+file_label)
    f_ens=np.load(file_label+'f_ensemble.npy')
    a_ens=np.load(file_label+'a_ensemble.npy')
    for i in range(50):
        comp_f=DistComparison(f_ens_b[i].T,f_ens[i].T)
        comp_a=DistComparison(a_ens_b[i].T,a_ens[i].T)
        kl_vals[i,0,k]=comp_f.compute_KL_with_weights()
        kl_vals[i,1,k]=comp_a.compute_KL_with_weights()
os.chdir('/home/shashank/Lorenz_63/Comparison_PF_Enkf')
np.save('kl_vals_fixed_bias={}_base={}.npy'.format(bias,lambda_),kl_vals)

# State and obs
# #obs=shorten(np.load('obs{}_gap_{}_H1__mu={}_obs_cov1.npy'.format((k),0.2,mu)),factor)

# #Go inside the data folder......................................
# file_label='bias=0.0_obs=2_ens=50_Mcov=0.1,ocov={}_,gap={}_alpha=1.0_loc=convex_r=0'.format(mu,ob_gap)
# os.chdir(file_label)
#
# #Load data....
# f_ens_b=np.load(file_label+'f_ensemble.npy')
# a_ens_b=np.load(file_label+'a_ensemble.npy') #ens has shape:=[time steps,system dimension,ensemble number]
# f_mean_b=np.sum(f_ens_b,axis=2)/N
# a_mean_b=np.sum(a_ens_b,axis=2)/N
# #time=np.load(file_label+'time.npy')

# kl_vals=np.zeros((50,2,9))
# for j in range(9):
#     os.chdir(str_r+'/ob{}'.format(j+2))
#     file_label='bias=10_obs=2_ens=50_Mcov=1,ocov={}_,gap={}_alpha=1.0_loc=convex_r=0'.format(mu,ob_gap)
#     os.chdir(file_label)
#     f_ens=np.load(file_label+'f_ensemble.npy')
#     a_ens=np.load(file_label+'a_ensemble.npy')
#     for i in range(50):
#         comp_f=DistComparison(f_ens_b[i].T,f_ens[i].T)
#         comp_a=DistComparison(a_ens_b[i].T,a_ens[i].T)
#         kl_vals[i,0,j]=comp_f.compute_KL_with_weights()
#         kl_vals[i,1,j]=comp_a.compute_KL_with_weights()
# os.chdir('/home/shashank/Lorenz_63/Trajectory_Observations')
# np.save('kl_vals_0.2.npy',kl_vals)
