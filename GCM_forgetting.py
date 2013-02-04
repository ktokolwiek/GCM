#!/usr/bin/python

# 29 January 2013
# This Python code is for modelling data from human categorisation experiments
# using Generalized Context Model (Nosofsky, 1986).

# 31 January 2013
# Need to think how to represent categories (self.catA and self.catB): whether
# this should be by unique instance number or by length or whatever. Porbably
# should then generate unique instance numbers. Should have a list of Ps which
# will contain all instance numbers for that participant.

# 1 February
# I just need to implement data output and incrementality in the model, so that
# category labels re-estimate based on the current model.

# 4 February
# Write to disk is ready-ish. Need a way to test and something to do with
# incrementality of the model. Precition of length from category label.

from random import random, gauss
from os import path
import math

class ps_data():

    def __init__(self, fname, verbose,\
            gamma, forget_rate, choice_parameter, noise_mu, noise_sigma):
        self.verbosity=verbose
        self.SetParameters(gamma, forget_rate, choice_parameter, noise_mu, noise_sigma)
        self.ReadData(fname)

    def __repr__(self):
        result=''
        try:
            result += ('Model with parameters: gamma: %(gamma)d, forget '+\
                    'rate: %(forget_rate)d, choice parameter: '+\
                    '%(choice_parameter)d, noise mean: %(noise_mu)d, '+\
                    'noise sd: %(noise_sigma)d\n'+\
                    'Data file used: %(datafile)s.\n') % self.__dict__
        except:
            result += 'Unparametrised GCM.\n'
        try:
            result += ('Re-estimated %(reEstimated)d instances. '+\
                    '%(changedCats)d instances changed class.') %self.__dict__
        except:
            result += 'No instances re-estimated.'
        return result

    # DATA INPUT:
    def ReadData(self, fname):
        """ Here we read in the data from file fname and will save it in the
        class instance's data structure."""
        self.datafile=path.join(path.realpath('.'),fname)
        self.data={} # The actual data structure
        #self.catA=[] # A list of members of category A
        #self.catB=[] # A list of members of category B
        # We don't put the data in categories yet - we do that in the modelling
        # phase, where we build the model incrementally.
        if path.exists(fname):
            with open(fname) as f:
                for line in f:
                    try:
                        a=line.split(',')
                        instNo=self.GetInstNo(a[0],a[3],a[1],a[2])
                        if a[5]=='D': # this is how it is represented in the
                            # data from Texas
                            actualcat=1
                            #self.catB.append(instNo)
                        else:
                            actualcat=-1
                            #self.catA.append(instNo)
                        # ^ this is the category which is given to the Ps as
                        # feedback. This is what we model as what they
                        # remember / forget / use for category inference.
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
                        self.AddPs(int(a[0]),int(a[3]),int(a[1]),int(a[2]),int(a[4]),\
                                actualcat, idealcat, pscat)
                    except:
                        continue # say, if first line or something wrong
        else:
            if self.verbose > 0:
                print "The filename "+fname+" is invalid!"
        pass

    def GetInstNo(self, ps_id, trial_no, session, condition):
        """Gets a supposedly unique number of a training instance"""
        return int('%09d%03d%02d%02d' % \
                (int(ps_id),int(trial_no),int(session),int(condition)))

    def AddPs(self, ps_id, trial_no, session, condition, length, actualCat,\
            idealCat, responseCat):
        """ Here we add in a datapoint for a participant. Can also use that for
        updating the Ps's data. """
        self.data[self.GetInstNo(ps_id,trial_no,session,condition)]={'ps_id':ps_id,\
                'trial_no': trial_no,\
                'session': session,\
                'condition': condition,\
                'length': self.AddNoise(length),\
                'actualCat': actualCat,\
                'idealCat': idealCat,\
                'responseCat': responseCat}

    def GetInstancesForPs(self, **kwargs):
        """Returns all instance ids for which the kwargs are matching with the
        instance."""
        result=[]
        try:
            for inst_id in self.data.keys():
                if all([self.data[inst_id][key]==kwargs[key]\
                        for key in kwargs.keys()]):
                    result.append(inst_id)
        except KeyError, e:
            if self.verbose > 0:
                print 'Error: keyword '+ e.message+' does not exist'

        return result

    # DATA OUTPUT:
    def GetPsData(self,inst_no):
        """Return the data for this trial"""
        try:
            return self.data[inst_no]
        except:
            if self.verbose > 10:
                print "No data for trial %d !" % inst_no
            return {'ps_id': 1, 'trial_no': 1, 'session': 1, 'condition': 1,\
                    'length': 1, 'actualCat':-1, 'idealCat': -1, 'responseCat': -1}

    def WriteOut(self,fname):
        """Writes out the data to the file fname"""
        with open(fname,'w') as f:
            f.write(self)
            f.write('ps_id\ttrial_no\tsession\tcondition\t'+\
                    'length\tactualCat\tidealCat\tresponseCat\t'+\
                    'predictedCat\n')
            for key in sorted(self.data.keys()):
                f.write(('(ps_id)\t(trial_no)\t(session)\t(condition)\t'+\
                        '(length)\t(actualCat)\t(idealCat)\t(responseCat)\t')%self.GetPsData(key)+\
                        self.PredictCategory(key)+'\n')

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
    def ReEstimateCategory(self,instance_no):
        """Re-estimate the category membership of the instance."""
        inst=self.GetPsData[instance_no]
        # delete number from its category - need to do that because else you
        # would use the category information of current instance to re-estimate
        # it.
        cat_old=inst['actualCat']
        inst['actualCat']=self.PredictCategory(instance_no)
        self.reEstimated += 1 # Add one to the number of re-estimated
        # instances
        if inst['actualCat']!=cat_old:
            self.changedCats +=1 # Add one instance to the
            # number of instances which changed category
            self.AddPs(**inst) # overwrites the datapoint
            if inst['actualCat']==1:
                self.catA.remove(instance_no)
                self.catB.append(instance_no)
            else:
                self.catB.remove(instance_no)
                self.catA.append(instance_no)

    def ReEstimateLength(self, instance_no):
        """Re-estimate the length of an instance based on category
        membership.
        The re-estimated length is the average length of stimuli in the
        category."""

        pass

    def ForgetInstances(self):
        """Selects instances which are to be forgotten, following an
        exponential distribution, based on recency of presentation."""
        self.reEstimated=0
        self.changedCats=0
        # Start counting from here. Possibly move it to a different place if I
        # decide to start counting from somewhere else, say while modelling
        # incrementally.
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
        return math.exp(-self.choice_parameter*math.sqrt((length1-length2)**2))

    def PredictCategory(self, instanceId):
        """Return the category membership of the instance.
        Assume categories are weighted equally. The gamma parameter is taken
        from the model parameter initialisation.
        Category A == -1
        Category B == 1"""
        if (len(self.catA)==0 and len(self.catB)==0):
            if self.verbose>0:
                print "Warning, trying to predict category on an "+\
                        "un-initialised model. Defaulting to a random "+\
                        "category."
            if random() < 0.5:
                return -1
            else:
                return 1
        else:
            psData=self.GetPsData(instanceId)['length']
            length=psData['length']
            catA=self.catA
            catB=self.catB
            if psData['actualCat']==-1:
                catA.remove(instanceId)
            else:
                catB.remove(instanceId)
            sum_cat_A=sum([self.Similarity(length, self.GetPsData(inst)['length']) for \
                inst in catA])
            sum_cat_B=sum([self.Similarity(length, self.GetPsData(inst)['length']) for \
                inst in catB])
            prob_a=sum_cat_A**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
            prob_b=sum_cat_B**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
            if prob_a>=prob_b:
                return -1
            else:
                return 1

    def TrainGCM(self):
        """Train the GCM incrementally using all instances."""
        # First, 

        pass

    def __init__(self):
        """For testing. """
        self.SetParameters(1,0.001,1,0,0.5)
        self.verbose=100
        # Add some instances.
        pass
