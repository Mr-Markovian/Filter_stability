import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import tensorflow as tf
from modules import wasserstein as tfw
import tensorflow_probability as tfp
tfd = tfp.distributions

# size of sample and realizations:
alpha2=1.0
sizes=np.array([50,100,200,400,800,1600])
realizations=20
dims=np.array([3])
distances=np.zeros((len(sizes),len(dims),realizations))
for k in range(len(sizes)):
    size=sizes[k]
    for i in range(len(dims)):
    #pdf=tfd.MultivariateNormalFullCovariance(loc =tf.zeros(dims[i]), covariance_matrix = tf.eye(dims[i]))
        pdf=tfd.MultivariateNormalTriL(loc=tf.zeros(dims[i]), scale_tril=tf.linalg.cholesky(alpha2*tf.eye(dims[i])))
        for j in range(realizations):
            data1=pdf.sample(size)
            data2=pdf.sample(size)
            loss=tfw.sinkhorn_loss(data1, data2, epsilon=0.01, num_iters=200, p=2)
            distances[k,i,j]=tf.sqrt(loss)

np.save('distances_sizes={}_to_{}_cov={}_realizations={}_dim={}_to_{}.npy'.format(sizes[0],sizes[-1],alpha2,realizations,dims[0],dims[-1]),distances)
    
