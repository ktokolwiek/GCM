#!/usr/bin/python
# 7 February
# Simple test case for the model.

import GCM_forgetting as GCM
import numpy as np
from multiprocessing import Pool


def testOneModel((gamma, forget_rate, choice_parameter, noise_mu,\
        noise_sigma)):
    """ Performs evaluation of one model with given parameters. Returns the log
    likelihood of the model fit.
    """
    fname = 'Randall_learn_lean_dataset.csv'
    model = GCM.ps_data(fname, 15, gamma, forget_rate, choice_parameter, \
            noise_mu, noise_sigma)
    instances = model.GetInstancesForPs(**{'condition': 1})
    model.PresentLoop(instances)
    return model.logLikelihood
    pass

def writeout(logLikelihoods, parameters, fname):
    """ Writes out the parameters and associated log likelihoods to file fname
    """
    with open(fname) as f:
        f.write('log_likelihood,gamma,forget_rate,choice_parameter,noise_mu,noise_sigma')
        for (ll, params) in zip(logLikelihoods, parameters):
            f.write('%.5f,%.5f,%.5f,%.5f,%.5f,%.5f' % ((ll,)+params))

def test():
    gammas=[0.5, 1, 2]
    forget_rates=np.logspace(-2,-.15, num=7)
    choice_parameters=[1,2]
    noise_mu=0
    noise_sigmas=[0.1,0.5]
    parameters = [(gamma, forget_rate, choice_parameter, noise_mu,\
        noise_sigma) for gamma in gammas for forget_rate in forget_rates\
        for choice_parameter in choice_parameters for noise_sigma\
        in noise_sigmas]
    pool=Pool(processes=20)
    logLikelihoods=pool.map(testOneModel, parameters)
    writeout(logLikelihoods, parameters, 'loglikelihoods.txt')
