import numpy as np
import os
import json

seed_num=45
np.random.seed(seed_num)

#specification of the experiments:
dim_x=40
m,n=20,40
forcing_x=8

# Choose the ensemble number to be generated.
N=400

#Now,load the initial condition,and integrate the ode:
with open('L_96_I_x0_on_attractor_dim_{},forcing_{}.json'.format(dim_x,forcing_x)) as jsonfile:
    parameters=json.load(jsonfile)

# parameters={}
# parameters['initial']=np.load('hidden_path.npy')[0]
mean_=parameters['initial']

#initial bias in the ensemble........
biases=np.array([0.0,2.0,4.0])

#initial ensemble spread
covs_=np.array([0.1,0.5,1.0])
#Initial_cov=parameters['lambda_']*np.eye(parameters['dim_x'])
seeds=np.array([11,13,17])

# Finalized experiments in April,2021
os.chdir(r'..')
os.mkdir('ensembles')
os.chdir(os.path.join(os.getcwd(),'ensembles'))
for i,j in zip(biases,covs_):
    "Select seeds such that there is a unique value for each set of complete parameters"
    Initial_cov=j*np.eye(parameters['dim_x'])
    x0_ensemble=np.random.multivariate_normal(np.asarray(mean_)+i,Initial_cov,N).T
    np.save('Ensemble={}_bias_{}_init_cov_{}_seed_{}.npy'.format(N,i,j,seed_num),x0_ensemble)
    #np.save('Ensemble={}_init_cov_{}_seed_{}.npy'.format(i,parameters['lambda_'],seed_num),x0_ensemble)

print('Job Done')

