#!/usr/bin/python

# 29 January 2013
# This Python code is for modelling data from human categorisation experiments
# using Generalized Context Model (Nosofsky, 1986).
from random import random, gauss
import math

class ps_data():

    def __init__(self,fname,verbose):
        self.data=[]
        self.verbosity=verbose
        self.ReadData(fname)

    # DATA INPUT:
    def ReadData(self, fname):
        """ Here we read in the data from file fname and will save it in the
        class instance's data structure."""
        with open(fname) as f:
            pass
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
    def SetParameters(self, gamma, forget_rate, choice_parameter,\
            noise_mu, noise_sigma):
        """
        Set the model's parameters:
        gamma parameter in GCM
        forgetting rate
        choice parameter
        noise ~ Gaussian(mean,sd).
        """
        self.gamma=gamma
        self.forget_rate=forget_rate
        self.choice_parameter=choice_parameter
        self.noise_mu=noise_mu
        self.noise_sigma=noise_sigma

    # FORGETTING
    def ReEstimateCategory(self,length):
        """Re-estimate the category membership of the instance."""
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
        """Return similarity measure between the two instances. Here the
        similarity measure is the exponential decay similarity function (cf.
        Maddox, 1999). We also use Euclidean distance as the measure of
        distance."""
        return math.exp(-self.choice_parameter*sqrt((length1-length2)**2))

    def PredictCategory(self, length):
        """Return the category membership of the instance.
        Assume categories are weighted equally. The gamma parameter is taken
        from the model parameter initialisation.
        Category A == -1
        Category B == 1"""
        sum_cat_A=sum([self.Similarity(length,lengthj) for lengthj in \
            self.catA])
        sum_cat_B=sum([self.Similarity(length,lengthj) for lengthj in \
            self.catB])
        prob_a=sum_cat_A**gamma/(sum_cat_A**gamma+sum_cat_B**gamma)
        prob_b=sum_cat_B**gamma/(sum_cat_A**gamma+sum_cat_B**gamma)
        if prob_a>=prob_b:
            return -1
        else:
            return 1

    def TrainGCM(self, instances):
        """Train the GCM using instances in the list"""
        pass


