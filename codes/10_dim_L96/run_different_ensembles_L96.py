"""The code here runs the Assimilation and saves the data in the desired folder with proper label
containing analysis ensemble,forecast ensemble,their respective mean,true state and the observations.
The 'data_visulaize.ipynb' file to load and view the results:
It calls Enkf_.py script.The parameters are supplied a value using a dictionary."""

from Enkf_ import *
from lorenz_96_ode import *
import numpy as np
import json
import os

# m1:dimension of observation Space ,n1:dimension of State Space
m1,n1=5,10
dim_x,forcing_x=n1,10
lambda_,mu=10,1.0

# Parameters to be controlled from here...
parameters={}
parameters['observables']=m1
parameters['lambda']     =lambda_
parameters['mu']         =mu
parameters['dim']        =n1         # Number of lattice points
parameters['N']          =50         # Ensemble Size
parameters['forcing_x']  =10          # forcing
parameters['obs_gap']    =0.1       # observation gap
parameters['alpha']      =1.0        # The inflation factor
parameters['assimilations']=400      #number of assimilations
parameters['loc']        =True       # True if localization is implemented,otherwise false
parameters['loc_fun']    ='gaspri'  #Type of function used for localization:'convex','concave','flat'
parameters['l_scale']    =4          #number of grid points choosen for localization

#The observation operator for measuring 20 alternate grid points:
H2_=np.zeros((m1,n1))

#Creating H2_
for i in range(m1):
    H2_[i,2*i]=1

#Creating a diagonal observation covariance matirx:
obs_cov1=mu*np.eye(m1)

#Setting parameters for Observation matrix,observation error-covariance
parameters['obs_cov']  =obs_cov1      # Observation error covariance matrix
parameters['H']        =H2_           # Observation operator

#Testing stability in lower dimension
biases=np.array([0.0,2.0,4.0])
covs_=np.array([0.1,0.5,1.0])


#suf_='gap={}/ocov={}'.format(parameters['obs_gap'],mu)
#------------------------Go to the desired folder-------------------------------
os.chdir(r'/home/shashank/Lorenz_63/10_dim_L96')
str1_=os.getcwd()
seeds=np.array([3,5,7,11,13,17,19,23,29,31])
# For different cases, change ob-gap and mu:
#iterate over observations:
for w in range(10):
    #np.random.seed(seeds[w])
    os.chdir(os.path.join(str1_,'ob{}'.format(w+1)))
    parameters['obs']=np.load('ob{}_gap_{}_H2_'.format(w+1,parameters['obs_gap'])+'_mu={}'.format(mu)+'_obs_cov1.npy')
    # iterate over the ensembles
    for i,j in zip(biases,covs_):
        #setting the seed for reproducibility
        parameters['lambda']=j
        Initial_cov1=parameters['lambda']*np.eye(parameters['dim'])
        parameters['model_cov']=Initial_cov1  # model error covariance matrix
        parameters['obs_cov']  =obs_cov1      # Observation error covariance matrix
        parameters['H']        =H2_
        # Load the ensembles from where they are saved:
        str_2=os.path.join(str1_,'ensembles')
        Ens=np.load(str_2+'/Ensemble={}_bias_{}_init_cov_{}_seed_{}.npy'.format(500,i,j,41))
        parameters['initial_ensemble']=Ens[:,:parameters['N']]
        parameters['initial']=np.mean(Ens[:,:parameters['N']],axis=1)
        #print(os.getcwd())
        #parameters['obs']=partial(np.load('ob{}_gap_{}_H1_'.format(w+1,0.2)+'_mu={}'.format(mu)+'_obs_cov1.npy'),n1-m1)
        os.chdir(os.path.join(str1_,'ob{}'.format(w+1)))
        obj=Enkf(parameters)
        obj.simulate()
        obj.label='bias={}_'.format(i)+obj.label
        #os.mkdir(obj.label)
        os.chdir(os.getcwd()+'/'+obj.label)
        obj.save_data()
        print('ob={},bias={},cov={}-run completed'.format(w+1,i,j))
        os.chdir(os.path.join(str1_,'ob{}'.format(w+1)))

#Setting parameters for Observation matrix,observation error-covariance and model error-covariance
# Creating an object with the above parameters specification
print('Job Done')


