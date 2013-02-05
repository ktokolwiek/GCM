#!/usr/bin/python

# 29 January 2013
# This Python code is for modelling data from human categorisation experiments
# using Generalized Context Model (Nosofsky, 1986).

# 31 January 2013
# Need to think how to represent categories (self.catA and self.catB): whether
# this should be by unique instance number or by length or whatever. Porbably
# should then generate unique instance numbers. Should have a list of Ps which
# will contain all instance numbers for that participant.

# 1 February 2013
# I just need to implement data output and incrementality in the model, so that
# category labels re-estimate based on the current model.

# 4 February 2013
# Write to disk is ready-ish. Need a way to test and something to do with
# incrementality of the model. Prediction of length from category label.

# 5 February 2013
# Write to disk now preserves the order of presentation. The ForgetLoop()
# function now incrementally presents the data.
# Possibly this is not a good idea, because then instances presented early get
# forgotten a few times...
# I also need to add a fit metric which will measure the fit of the model to
# the data with current parameters.

from random import random, gauss
from os import path
import sys,math

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
                    'rate: %(forget_rate).5f, choice parameter: '+\
                    '%(choice_parameter)d, noise mean: %(noise_mu).5f, '+\
                    'noise sd: %(noise_sigma).5f\n'+\
                    'Data file used: %(datafile)s.\n') % self.__dict__
        except:
            result += 'Unparametrised GCM.\n'
        try:
            result += ('Re-estimated %(reEstimated)d instances. '+\
                    '%(changedCats)d instances changed class.\n') %self.__dict__
        except:
            result += 'No instances re-estimated.\n'
        return result

    # DATA INPUT:
    def ReadData(self, fname):
        """ Here we read in the data from file fname and will save it in the
        class instance's data structure."""
        self.datafile=path.join(path.realpath('.'),fname)
        self.data={} # The actual data structure
        self.catA=[] # A list of members of category A
        self.catB=[] # A list of members of category B
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
                        self.AddPs(**{'ps_id': int(a[0]),\
                                'trial_no': int(a[3]),\
                                'session': int(a[1]),\
                                'condition': int(a[2]),\
                                'length': self.AddNoise(int(a[4])),\
                                #Adds noise only once
                                'actualCat': actualcat,\
                                'idealCat': idealcat,\
                                'responseCat': pscat})
                    except:
                        continue # say, if first line or something wrong
        else:
            if self.verbose > 0:
                print "The filename "+fname+" is invalid!"
        pass

    def GetInstNo(self, ps_id, trial_no, session, condition):
        """Gets a supposedly unique number of a training instance"""
        return '%09d%02d%04d%02d' % \
                (int(ps_id),int(session),int(trial_no),int(condition))

    def AddPs(self, **kwargs):
        """ Here we add in a datapoint for a participant. Can also use that for
        updating the Ps's data. """
        self.data[self.GetInstNo(kwargs['ps_id'],kwargs['trial_no'],\
                kwargs['session'],kwargs['condition'])]=kwargs

    def GetInstancesForPs(self, **kwargs):
        """Returns all instance ids for which the kwargs are matching with the
        instance."""
        result=[]
        for inst_id in self.data.keys():
            try:
                if all([self.data[inst_id][key]==kwargs[key]\
                        for key in kwargs.keys()]):
                    result.append(inst_id)
            except KeyError, e:
                if self.verbose > 0:
                    print 'Error: keyword '+ e.message+' does not exist.'
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
            f.write(self.__repr__())
            f.write('ps_id\ttrial_no\tsession\tcondition\t'+\
                    'length\tactualCat\tidealCat\tresponseCat\t'+\
                    'modelledCat\n')
            for key in sorted(self.data.keys()):
                try:
                    f.write(('%(ps_id)s\t%(trial_no)s\t%(session)s\t'+\
                            '%(condition)s\t%(length)s\t%(actualCat)s\t'+\
                            '%(idealCat)s\t%(responseCat)s\t%(modelledCat)s\n')\
                            % self.GetPsData(key))
                except:
                    f.write(('%(ps_id)s\t%(trial_no)s\t%(session)s\t'+\
                            '%(condition)s\t%(length)s\t%(actualCat)s\t'+\
                            '%(idealCat)s\t%(responseCat)s\t')\
                            %self.GetPsData(key) +\
                            str(self.PredictCategory(key))+'\n')

    def GraphCategories(self):
        """ Makes a graph of category distribution.
        """
        from matplotlib import pyplot
        import numpy as np
        # Presented distribution
        pyplot.subplot(411)
        As=[self.GetPsData(i)['length'] for i in \
                self.GetInstancesForPs(**{'actualCat': -1})]
        Bs=[self.GetPsData(i)['length'] for i in \
                self.GetInstancesForPs(**{'actualCat': 1})]
        n, bins, patches = pyplot.hist([As,Bs], bins=60)
        mean_a=np.mean(As)
        sd_a=np.std(As)
        mean_b=np.mean(Bs)
        sd_b=np.std(Bs)
        pyplot.title(('Presented distribution. A: mean %.2f sd %.2f; B: mean'+\
                ' %.2f sd %.2f.')%(mean_a,sd_a,mean_b,sd_b))
        # Initial responses
        pyplot.subplot(412)
        As=[self.GetPsData(i)['length'] for i in \
                self.GetInstancesForPs(**{'session': 1,\
                'responseCat': -1})]
        Bs=[self.GetPsData(i)['length'] for i in \
                self.GetInstancesForPs(**{'session': 1,\
                'responseCat': 1})]
        n, bins, patches = pyplot.hist([As,Bs], bins=60)
        mean_a=np.mean(As)
        sd_a=np.std(As)
        mean_b=np.mean(Bs)
        sd_b=np.std(Bs)
        pyplot.title(('Initial responses. A: mean %.2f sd %.2f; B: mean'+\
                ' %.2f sd %.2f.')%(mean_a,sd_a,mean_b,sd_b))
        # Post - forgetting
        pyplot.subplot(413)
        As = [self.GetPsData(i)['length'] for i in self.catA]
        Bs = [self.GetPsData(i)['length'] for i in self.catB]
        n, bins, patches = pyplot.hist([As,Bs], bins=60)
        mean_a=np.mean(As)
        sd_a=np.std(As)
        mean_b=np.mean(Bs)
        sd_b=np.std(Bs)
        pyplot.title(('Post-forgetting. A: mean %.2f sd %.2f; B: mean'+\
                ' %.2f sd %.2f.')%(mean_a,sd_a,mean_b,sd_b))
        # Responses in test session
        # FIXME: use actual test session data
        pyplot.subplot(414)
        As=[self.GetPsData(i)['length'] for i in \
                self.GetInstancesForPs(**{'session': 5,\
                'responseCat': -1})]
        Bs=[self.GetPsData(i)['length'] for i in \
                self.GetInstancesForPs(**{'session': 5,\
                'responseCat': 1})]
        n, bins, patches = pyplot.hist([As,Bs], bins=60)
        mean_a=np.mean(As)
        sd_a=np.std(As)
        mean_b=np.mean(Bs)
        sd_b=np.std(Bs)
        pyplot.title(('Test session. A: mean %.2f sd %.2f; B: mean'+\
                ' %.2f sd %.2f.')%(mean_a,sd_a,mean_b,sd_b))
        pyplot.show()

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
        if forget_rate==0.0:
            self.forget_rate=1e-50
            # so that we don't get division by zero, but still (virtually) no
            # forgetting
        else:
            self.forget_rate=forget_rate
        self.choice_parameter=choice_parameter
        self.noise_mu=noise_mu
        self.noise_sigma=noise_sigma

    # FORGETTING
    def ForgetOnce(self):
        """Selects instances which are to be forgotten, following an
        exponential distribution, based on recency of presentation."""
        forgetCatA=[(math.exp(-(1.0/self.forget_rate)*(i+1)), no) \
                for (i,no) in enumerate(self.catA)]
        forgetCatB=[(math.exp(-(1.0/self.forget_rate)*(i+1)), no) \
                for (i,no) in enumerate(self.catB)]
        for (prob, instNo) in forgetCatA:
            if random() < prob:
                self.ReEstimateCategory(instNo)
        for (prob, instNo) in forgetCatB:
            if random() < prob:
                self.ReEstimateCategory(instNo)

    def ForgetLoop(self, instances):
        """ Loops through the instances in the list and after presenting every
        instance goes through the cycle of forgetting.
        If the list is None, then defaults to presenting all instances."""
        if not instances:
            instances = sorted(self.data.keys())
        if self.verbose > 10:
            print "Forgetting loop"
            sys.stdout.write("[%s]" % (" " * 20))
            sys.stdout.flush()
            sys.stdout.write("\b" * (20+1))
        for (i,instId) in enumerate(instances):
            inst=self.GetPsData(instId)
            try:
                cat=inst['modelledCat']
            except:
                cat=inst['actualCat']
            if cat==-1:
                self.catA.append(instId)
            else:
                self.catB.append(instId)
            self.ForgetOnce()
            if self.verbose > 10:
                if (i % (len(self.data.keys())/20)) == 0:
                    sys.stdout.write("-")
                    sys.stdout.flush()
        if self.verbose > 10:
            sys.stdout.write("\n")

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

    def ReEstimateCategory(self,instance_no):
        """Re-estimate the category membership of the instance."""
        inst=self.GetPsData(instance_no)
        # delete number from its category - need to do that because else you
        # would use the category information of current instance to re-estimate
        # it.
        try:
            cat_old=inst['modelledCat'] # If the instance was already
            # re-estimated using the GCM.
        except:
            cat_old=inst['actualCat'] # If it has not been modelled before
        inst['modelledCat']=self.PredictCategory(instance_no)
        try:
            self.reEstimated += 1 # Add one to the number of re-estimated
        except:
            self.reEstimated = 1
        # instances
        if inst['modelledCat']!=cat_old:
            try:
                self.changedCats +=1 # Add one instance to the
                # number of instances which changed category
            except:
                self.changedCats = 1
            self.AddPs(**inst) # overwrites the datapoint
            if cat_old==-1:
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

    def PredictCategory(self, instanceId):
        """Return the category membership of the instance.
        Assume categories are weighted equally. The gamma parameter is taken
        from the model parameter initialisation.
        Category A == -1
        Category B == 1"""
        if (len(self.catA)<=1 and len(self.catB)<=1):
            if self.verbose>0:
                print "Warning, trying to predict category on an "+\
                        "un-initialised model. Defaulting to a random "+\
                        "category."
            if random() < 0.5:
                return -1
            else:
                return 1
        else:
            psData=self.GetPsData(instanceId)
            length=psData['length']
            try:
                cat=psData['modelledCat']
            except:
                cat=psData['actualCat'] # If the instance has not been
                # re-estimated
            sum_cat_A=sum([self.Similarity(length, self.GetPsData(inst)['length']) for \
                inst in self.catA if inst != instanceId])
            sum_cat_B=sum([self.Similarity(length, self.GetPsData(inst)['length']) for \
                inst in self.catB if inst != instanceId])
            prob_a=sum_cat_A**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
            prob_b=sum_cat_B**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
            if prob_a>=prob_b:
                return -1
            else:
                return 1

    def TrainGCM(self):
        """Train the GCM incrementally using all instances."""
        # First, present the instances to the model one by one.
        # Next, estimate the model response.
        # Then forget and re-estimate
        pass

    def __init__(self):
        """For testing. """
        self.SetParameters(1,0.7,1,0,0.5)
        self.verbose=100
        # Add some instances.
        pass
