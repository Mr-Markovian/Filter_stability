import numpy as np
import os
import json
os.chdir('..')
seed_num=41
np.random.seed(seed_num)

#specification of the experiments:
dim_x=10
m,n=5,10
forcing_x=10

# Choose the ensemble number to be generated.
N_ens=500

#Now,load the initial condition,and integrate the ode:
with open('L_96_I_x0_on_attractor_dim_{},forcing_{}.json'.format(dim_x,forcing_x)) as jsonfile:
    parameters=json.load(jsonfile)

# parameters={}
# parameters['initial']=np.load('hidden_path.npy')[0]
mean_=np.load('trajectory_500.npy')[0]

#initial bias in the ensemble........
biases=np.array([0.0,2.0,4.0])

#initial ensemble spread
covs_=np.array([0.1,0.5,1.0])
#Initial_cov=parameters['lambda_']*np.eye(parameters['dim_x'])

# Finalized experiments in April,2021
os.mkdir('ensembles')
os.chdir(os.path.join(os.getcwd(),'ensembles'))
for i,j in zip(biases,covs_):
    "Select seeds such that there is a unique value for each set of complete parameters"
    Initial_cov=j*np.eye(dim_x)
    x0_ensemble=np.random.multivariate_normal(np.asarray(mean_)+i,Initial_cov,N_ens).T
    np.save('Ensemble={}_bias_{}_init_cov_{}_seed_{}.npy'.format(N_ens,i,j,seed_num),x0_ensemble)
    #np.save('Ensemble={}_init_cov_{}_seed_{}.npy'.format(i,parameters['lambda_'],seed_num),x0_ensemble)

print('Job Done')

