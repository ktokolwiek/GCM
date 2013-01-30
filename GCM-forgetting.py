#!/usr/bin/python

# 29 January 2013
# This Python code is for modelling data from human categorisation experiments
# using Generalized Context Model (Nosofsky, 1986).
from random import random, gauss

class ps_data():

    def __init__(self,fname,verbose):
        self.data=[]
        self.verbosity=verbose
        self.ReadData(fname)

    # DATA INPUT:
    def ReadData(self, fname):
        """ Here we read in the data from file fname and will save it in the
        class instance's data structure."""
        pass

    def AddPs(self, ps_id, trial_no, session, condtiion, length, actualCat,\
            idealCat, responseCat):
        """ Here we add in a datapoint for a participant """
        pass

    # DATA OUTPUT:
    def GetPsData(self,ps_id,trial_no):
        """Return the data for this participant"""
        pass

    def WriteOut(self,fname):
        """Writes out the data to the file fname"""
        pass

    # MODEL
    def SetParameters(self, decision_rate, forget_rate, choice_parameter,\
            noise_mu, noise_sigma):
        """
        Set the model's parameters:
        decision rate
        forgetting rate
        choice parameter
        noise ~ Gaussian(mean,sd).
        """
        self.decision_rate=decision_rate
        self.forget_rate=forget_rate
        self.choice_parameter=choice_parameter
        self.noise_mu=noise_mu
        self.noise_sigma=noise_sigma

    # FORGETTING
    def ReEstimateCategory(self,ps_id,trial_no):
        """Re-estimate the category membership of the instance """
        pass

    # PERCEPTUAL NOISE
    def AddNoise(self, length):
        """
        Add perceptual noise.
        Controls for the case when perceptual noise is a negative number and we
        get <0 (returns 0 in such case).
        """
        return max(0,length+gauss(self.noise_mu, self.noise_sigma))

    # GCM
    def Similarity(self, length1, length2):
        """Return similarity measure between the two instances"""
        pass

    def PredictCategory(self, length):
        """Return the category membership of the instance"""
        pass

    def TrainGCM(self, instances):
        """Train the GCM using instances in the list"""
        pass


