#!/usr/bin/python

# 29 January 2013
# This Python code is for modelling data from human categorisation experiments
# using Generalized Context Model (Nosofsky, 1986).

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

    # FORGETTING
    def ReEstimateCategory(self,ps_id,trial_no):
        """Re-estimate the category membership of the instance """
        pass

    # MODEL
    def Similarity(self, ps_id1,trial_no1, ps_id2,trial_no2):
        """Return similarity measure between the two instances"""
        pass


