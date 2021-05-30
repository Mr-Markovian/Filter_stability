import numpy as np
# The way we did the assimilation , the same loops 
mu=0.1       #1.0  #observation covariance
ob_gap=0.1   # observation gap 0.1,0.2,0.3

#initial bias in the ensemble....
biases=np.array([0.0,3.0,6.0])
covs_=np.array([0.01,0.1,1.0])

suf_='gap={}/ocov={}'.format(ob_gap,mu)
#iterate over observations:
for w in range(10):
    # go inside a particular observation,obs_gap and ocov.
    os.chdir(os.path.join(str1_,'ob{}'.format(w+1),suf_))
    for i in range(len(biases)):
        bias=biases[i]
        for j in range(len(covs_)):
            #setting the seed for reproducibility
            seed_num=int(47*(i+1)*(j+1))
            np.random.seed(seed_num)
            #print(os.getcwd())
            #Load the experiment


            #print('ob={},bias={},cov={}-run completed'.format(w+1,biases[i],covs_[j]))
            os.chdir(os.path.join(str1_,'ob{}'.format(w+1),suf_))
#Setting parameters for Observation matrix,observation error-covariance and model error-covariance
# Creating an object with the above parameters specification
print('Job Done')
