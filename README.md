
This repository contains a part of data and codes related to experiments performed on Lorenz-96 chaotic system for dimensions 10 and 40 using EnKF( Ensemble Kalman Filter). The experiments use EnKF to compute the forecast and analysis ensemble. The assimilated experiments data are not uploaded but the codes to recreate the experiments are avaiable. The final results of distance calculation is present. 

## Filters
Filters are state( and parameter) estimation algorithms which produce conditional estimates and distributions of the state of a dynamical system based on a numerical model and observations from the real system which are better than either of the model estimate and the observations. Two generic filters used in the literature are ensemble kalman filters and particle filters. The former uses unweighted ensemble and approximates the kalman filter update equations and the latter uses weihted particles and sequential sampling of wights.  

## Distance between probability distributions falling over time-> Stability
Stability of a filter a measure of how different initial distribution of the filter lead to similar conditional distribution over assimilation time.
Understanding how two different ensemble representing conditional distribution converge( in probability) is important from both theoretical and practical applications. A good measure of distance on probability distribution which became recently efficient to compute is Sinkhorn divergence, a good proxy of wasserstein distance on the space of probability distribution, which is derived from the idea of Optimal Transport.

The project demostrates how to numerically compute the distance between conditional distributions of the state over time. The idea is to understand and explore filter stability of the two differet general purpose filters- Ensemble Kalma Filter and Particle Filters. The publication which came out of this study is availabel here [Stability of nonlinear filters - numerical explorations of particle and ensemble Kalman filters](https://ieeexplore.ieee.org/document/9703185/)
