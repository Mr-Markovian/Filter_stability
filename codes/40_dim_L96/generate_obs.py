"""Load the true trajectory and create the observations:"""
import numpy as np
import os

# Code for shortening the object in time :
def shorten(arr, factor=10):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/factor)
    new_arr = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        new_arr[i] = arr[i*factor]
    return new_arr

#np.random.seed(54)
#Observation Operator and noise statistics need to be defined first:
dim_x=40
m,n=20,40
forcing_x=8
mu=1.0
factor_=1
obs_gap=0.1

#Observation operator to observe consecutive grid points :
H1_=np.zeros((m,n))
H1_[:m,:m]=np.eye(m)

#Observation operator for alternate grid points:
H2_=np.zeros((m,n))
for i in range(m):
    H2_[i,2*i]=1

obs_cov1=mu*np.eye(m)
os.chdir('..')
#For lorenz I level system:
file_name='Trajectory_{}_seed_{}.npy'.format(0.1,35)
#Load the trajectory (a .npy file):
State=shorten(np.load(file_name),factor=factor_)
seeds=np.array([3,5,7,11,13,17,19,23,29,31,37])

str1_=os.getcwd()
for i in range(10):
    os.chdir(str1_)
    os.mkdir('ob{}'.format(i+1))
    os.chdir(str1_+'/ob{}'.format(i+1))
    np.random.seed(seeds[i])
    obs=(H2_@(State.T)).T+np.random.multivariate_normal(np.zeros(m),obs_cov1,len(State[:,0]))
    np.save('ob{}_gap_{}_H2_'.format(i+1,round(obs_gap*factor_,2))+'_mu={}'.format(mu)+'_obs_cov1.npy',obs)
print('Job Done')
