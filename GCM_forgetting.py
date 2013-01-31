#!/usr/bin/python

# 29 January 2013
# This Python code is for modelling data from human categorisation experiments
# using Generalized Context Model (Nosofsky, 1986).

# 31 January 2013
# Need to think how to represent categories (self.catA and self.catB): whether
# this should be by unique instance number or by length or whatever. Porbably
# should then generate unique instance numbers. Should have a list of Ps which
# will contain all instance numbers for that participant.
from random import random, gauss
from os import path
import math

class ps_data():

    def __init__(self,fname,verbose):
        self.verbosity=verbose
        self.ReadData(fname)

    # DATA INPUT:
    def ReadData(self, fname):
        """ Here we read in the data from file fname and will save it in the
        class instance's data structure."""
        self.data={}
        self.catA=[]
        self.catB=[]
        if path.exists(fname):
            with open(fname) as f:
                for line in f:
                    try:
                        a=line.split(',')
                        if a[5]=='D':
                            actualcat=1
                            self.catB.append(int(a[4]))
                        else:
                            actualcat=-1
                            celf.catA.append(int(a[4]))
                        # ^ this is the category which is given to the Ps as
                        # feedback.
                        pscat=-1
                        if a[6]=='D':
                            pscat=1
                        # ^ this is the category the Ps responded. Most relevant in
                        # test set.
                        idealcat=-1
                        if int(a[4])>30:
                            idealcat=1
                        # ^ this is the category which the ideal classifier
                        # would put the stimulus in.
                        self.AddPs(int(a[0]),int(a[0]+a[3]),int(a[1]),int(a[2]),int(a[4]),\
                                actualcat, idealcat, pscat)
                    except:
                        continue # say, if first line or something wrong
        else:
            if verbose > 0:
                print "The filename "+fname+" is invalid!"
        pass

    def AddPs(self, ps_id, trial_no, session, condition, length, actualCat,\
            idealCat, responseCat):
        """ Here we add in a datapoint for a participant """
        try:
            self.data[ps_id][trial_no]={'session': session,\
                    'condition': condition,\
                    'length': length,\
                    'actualcat': actualCat,\
                    'idealcat': idealCat,\
                    'reponsecat': responseCat}
        except:
            self.data[ps_id]={trial_no:\
                    {'session': session,\
                    'condition': condition,\
                    'length': length,\
                    'actualcat': actualCat,\
                    'idealcat': idealCat,\
                    'reponsecat': responseCat}}

    # DATA OUTPUT:
    def GetPsData(self,ps_data):
        """Return the data for this participant"""
        try:
            return self.data[ps_data]
        except:
            if verbose > 100:
                print "No data for participant %d and trial %d !" % \
                {ps_id,trial_no}
            return {1:{'session': 1, 'condition': 1, 'length': 1, 'actualcat':-1,\
                    'idealcat': -1, 'responsecat': -1}}

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
    def ReEstimateCategory(self,ps_id,trial_no):
        """Re-estimate the category membership of the instance."""
        self.data[ps_id][trial_no]
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


