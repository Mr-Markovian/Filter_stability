"""This script implements the Ensemble kalamn Filter algorithm by defining the class Enkf
which inherits the model class of the respective system(ode or pde)."""

"""The Five crucial sub-parts of the code are:
1)Initialize: Generate the initial ensemble at time t=0
2)Assimilate: Assimilate the observation at present(defined as obs_gap)
3)Forecast: Integrate to get the analysis to next step and the observation and observation perturbations
4)simulate: Calls all the above functions to perform the respective tasks"""

#Later parts of the code has options to plot various things and were used
#while writing the code in jupyter-notebook.Not much useful once we start saving and visualizing data

from lorenz_96_ode import *
import numpy as np

# A bit of speeding up...
zo=np.zeros
zo_l=np.zeros_like
mvr=np.random.multivariate_normal
ccat=np.concatenate
p_inv=np.linalg.pinv
abs=np.absolute

class Enkf(lorenz_96):
    def __init__(self,parameters):
        """Class attributes definition and initialization"""
        # parse parameters
        for key in parameters:
            setattr(self, key, parameters[key])
        self.final        =self.initial
        #self.forecast_mean=zo((1,self.dim))                # To store the predicted value of the state
        #self.analysis_mean=zo((1,self.dim))                # to store the estimate of the true State
        self.analysis_ensemble=zo((self.assimilations,self.dim,self.N))     # to store the analyized Ensemble
        self.forecast_ensemble=zo((self.assimilations+1,self.dim,self.N))     # to store the forecast Ensemble
        #self.innov_   =zo((1,self.observables,self.N))     # storing the innovation vector
        self.A_p      =zo((self.dim,self.N))
        self.A        =self.initial_ensemble
        self.D_p      =zo((self.observables,self.N))
        #self.analysis_mean[0]    =self.initial
        #self.forecast_mean[0]    =self.initial
        self.analysis_ensemble[0]=self.initial_ensemble
        self.forecast_ensemble[0]=self.initial_ensemble
        #self.time=self.obs_gap*np.arange(0,self.assimilations)

        # The label created by using the most important set of parameters used to create the Object
        self.label    ='obs={}_ens={}_Mcov={},ocov={}_,gap={}_alpha={}_loc={}_r={}'.format(parameters['observables'],parameters['N'],parameters['lambda'],parameters['mu'],parameters['obs_gap'],parameters['alpha'],parameters['loc_fun'],parameters['l_scale'])
        # if localization is implemented,create the localization matrix 'self.rho'
        if (self.loc):
            if (self.loc_fun=='convex'):
                f1=lambda r:0 if r>self.l_scale-1 else np.round(2*np.exp(-r/self.l_scale)/(1+np.exp(r)),3)
                self.rho=np.asarray([[f1(min(abs(i-j),abs(self.dim-abs(i-j)))) for j in range(self.dim)]for i in range(self.dim)])
            if (self.loc_fun=='concave'):
                f2=lambda r:0 if r>self.l_scale else np.round((self.l_scale**2-r*r)/self.l_scale**2,3) # Concave
                self.rho=np.asarray([[f2(min(abs(i-j),abs(self.dim-abs(i-j)))) for j in range(self.dim)]for i in range(self.dim)])
            if (self.loc_fun=='flat'):
                f3=lambda r:0 if r>self.l_scale else np.round((self.l_scale-r)/self.l_scale,3)
                self.rho=np.asarray([[f3(min(abs(i-j),abs(self.dim-abs(i-j)))) for j in range(self.dim)]for i in range(self.dim)])
            if (self.loc_fun=='gaspri'):
                def f5(r):
                #Defining the function as given in Alberto-Carassi DA-review paper
                    r_=abs(r)/self.l_scale
                    if 0.<=r_ and r_<1.:
                        return 1.-(5./3)*r_**2+(5./8)*r_**3+(1./2)*r_**4-(1./4)*r_**5
                    elif 1.<=r_ and r_<2.:
                        return 4.-5.*r_+(5./3.)*r_**2+(5./8.)*r_**3-(1./2.)*r_**4+(1/12)*r_**5-2/(3*r_)
                    else:
                        return 0
                self.rho=np.asarray([[f5(min(abs(i-j),abs(self.dim-abs(i-j)))) for j in range(self.dim)]for i in range(self.dim)])


    def forecast(self):
        "Integrate the ensemble until new observation"
        "Save the forcast upto initial time+obs_gap"
        self.time_start=self.obs_gap*(self.i) #i
        self.time_stop=self.obs_gap*((self.i)+1)    #i+1
        self.t_evaluate=None
        for j in range(self.N):
            self.initial   =self.A[:,j]
            self.A[:,j]    =self.solution()[-1]

        #self.forecast_ensemble=ccat((self.forecast_ensemble,[self.A]),axis=0)
        self.forecast_ensemble[(self.i)+1]=self.A
        mean                  =np.sum(self.A,axis=1)/(self.N)
        #self.forecast_mean    =ccat((self.forecast_mean,[mean]),axis=0)
        # To compute and the ensemble perturbations and use multiplicative inflation
        self.A_p              =self.alpha*(self.A-np.tile(mean,(self.N,1)).T)

    def analysis(self):
        """All computation for analysis after observation"""

        obs_t=self.obs[self.i] #i+1->i
        self.D           =mvr(obs_t,self.obs_cov,self.N).T
        temp             =np.sum(self.D,axis=1)/(self.N)  # temporary variable to store the mean

        # To compute and store innovation vectors
        self.innovation   =self.D-self.H@self.A

        """use k_gain=(P_f)(H)^t[H P_f H^t + R ]^(-1) if localization is implemented"""
        p_f  =self.A_p@np.transpose(self.A_p)/(self.N-1)
        if (self.loc):   # If localization is implemented,replace p_f by schur product of self.rho and p_f
            p_f =self.rho*p_f
        p_f_h_t =p_f@np.transpose(self.H)

        # Compute the kalman gain matrix:
        k_gain=p_f_h_t@p_inv(self.H@p_f_h_t+self.obs_cov)

        # The analysis ensemble, used in the next time for integration:
        self.A            =self.A+k_gain@self.innovation

        # The analysis mean
        mean              =np.sum(self.A,axis=1)/(self.N)
        #self.innov_       =ccat((self.innov_,[self.innovation]),axis=0)
        #self.analysis_mean=ccat((self.analysis_mean,[mean]),axis=0)

        # Storing the analysis ensemble at current time
        self.analysis_ensemble[self.i]=self.A
        #self.analysis_ensemble=ccat((self.analysis_ensemble,[self.A]),axis=0)

    def simulate(self):
        """Time-total is number of Assimilations performed"""
        for self.i in range(0,self.assimilations):
            self.analysis()
            self.forecast()
        #self.analysis_ensemble=np.delete(self.analysis_ensemble,0,axis=0)
        self.forecast_ensemble=np.delete(self.forecast_ensemble,-1,axis=0)

    #def reset():
    #    """Reset the object to initial configuration.Note that this trick works if all the class variables
    #    are defined inside __init__ function"""
    #    self.__init__()

    def save_data(self):
        "to save forecast,analysis ensemble with time"
        np.save(self.label+'f_ensemble.npy',self.forecast_ensemble)
        np.save(self.label+'a_ensemble.npy',self.analysis_ensemble)
        #np.save(self.label+'time.npy',self.time)
