#!/usr/bin/python

from random import random, gauss
from os import path
import sys, math, time, csv
import numpy as np

class ps_data():

    def __init__(self, fname, verbose=15, gamma=1, forget_rate=0.0001,\
            choice_parameter=1, noise_mu=0, noise_sigma=0.5):
        self.verbose=verbose
        self.SetParameters(gamma, forget_rate, choice_parameter, noise_mu, noise_sigma)
        self.ReadData(fname)
        self.SetInstances()

    def __repr__(self):
        result=''
        try:
            result += ('#Model with parameters: gamma: %(gamma)d, forget '+\
                    'rate: %(forget_rate).5f, choice parameter: '+\
                    '%(choice_parameter)d, noise mean: %(noise_mu).5f, '+\
                    'noise sd: %(noise_sigma).5f\n'+\
                    '#Data file used: %(datafile)s.\n') % self.__dict__
        except:
            result += '#Unparametrised GCM.\n'
        try:
            result += '#Log-likelihood: %.5f\n' % (self.logLikelihood)+\
                    ('#Re-estimated %(reEstimated)d instances. '+\
                    '%(changedCats)d instances changed class.\n') %self.__dict__
        except:
            result += '#No instances re-estimated.\n'
        return result

    def SetInstances(self, instances=None, dataset='training'):
        """ Sets the instances which we used in the current model.
        """
        if not instances:
            if dataset == 'training':
                self.trainingInstances = sorted(self.instanceMap.values())
                self.usedTrainingData = self.trainingData.copy()
            else:
                self.testInstances = self.testData.keys()
                self.usedTestData = self.testData.copy()
        else:
            if dataset == 'training':
                self.trainingInstances = instances
                self.usedTrainingData = {key: self.trainingData[key] for key\
                        in instances}
            else:
                self.testInstances = instances
                self.usedTestData = {key: self.testData[key] for key\
                        in instances}

    def PrecomputeSimilarities(self):
        from sklearn.metrics.pairwise import euclidean_distances
        if self.verbose > 10:
            print 'Precomputing similarities...'
        X=np.matrix(self.usedTrainingData['length']).transpose()
        self.Similarities = \
                np.exp(-self.choice_parameter*euclidean_distances(X))


    # DATA INPUT:
    def ReadData(self, fname):
        """ Here we read in the data from file fname and will save it in the
        class instance's data structure."""
        self.datafile=path.join(path.realpath('.'),fname)
        self.logLikelihood=0
        self.instanceMap = {}
        datatype=np.dtype([('ps_id', int),('trial_no', int),\
                ('session', int),('condition', int),('length',float),\
                ('actualCat',int),('idealCat',int),('responseCat',int),\
                ('modelledCat', int)])
        data=[] # A temporary data structure which we are using before we
        # transfer all to numpy.
        self.testData = {} # initialise it here
        self.usedTestData = {} # The data structure we use in the code, with
        # selected instances
        self.catA=set() # Boolean saying that we don't have members in
        # category A
        self.catB=set() # same for cat B
        self.reEstimated=0
        self.changedCats = 0
        self.presentedOrder=[] # This will remember the order of presentation
        # of stimuli, for modelling of recency and forgetting. 
        # We don't put the data in categories yet - we do that in the modelling
        # phase, where we build the model incrementally.
        if path.exists(fname):
            with open(fname) as f:
                for line in f:
                    try:
                        a=line.split(',')
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
                        data.append((int(a[0]), int(a[3]), int(a[1]), int(a[2]),\
                                self.AddNoise(int(a[4])), actualcat, idealcat,\
                                pscat, actualcat))
                        # Here we are using actualCat as if it was modelledCat,
                        # to simplify choosing instances which are presented in
                        # category A or B.
                        # ps_id, trial_no, session, condition, length,
                        # actualCat, idealCat, responseCat, modelledCat
                    except Exception, e:
                        continue # say, if first line or something wrong
            self.trainingData = np.array(data,dtype=datatype)
            self.usedTrainingData = self.trainingData.copy()
            # Populate instanceMap
            for i, instance in enumerate(data):
                self.instanceMap[instance[0:4]]=i
        else:
            if self.verbose > 0:
                print "The filename "+fname+" is invalid!"

    def ReadTestData(self, fname):
        """ We read in test data.
        """
        self.testData={}
        if path.exists(fname):
            with open(fname) as f:
                reader=csv.DictReader(f)
                for line in reader:
                    if line['responseCat']=='A':
                        line['responseCat']=-1
                    else:
                        line['responseCat']=1
                    self.AddTestPs(**{'ps_id': int(line['ps_id']),\
                            'trial_no': int(line['trial_no']),\
                            'session': int(line['session']),\
                            'condition': int(line['condition']),\
                            'length': self.AddNoise(int(line['length'])),\
                            'responseCat': int(line['responseCat'])})
        else:
            if self.verbose > 0:
                print "The test file "+fname+" doesn't exist!"

    def GetInstNo(self, ps_id, trial_no, session, condition):
        """Gets a supposedly unique number of a training instance"""
        instanceTuple = (ps_id,session,trial_no,condition)
        try:
            # If the instance is in our mapping already
            result = self.instanceMap[instanceTuple]
        except KeyError:
            # It is not, so we need to add it as the next number and also
            # expand the data structure.
            self.instanceMap[instanceTuple]=result
        return result

    def AddTestPs(self, **kwargs):
        """ We add a participant's test datapoint, can also be used to update
        the instance data."""
        self.testData[self.GetInstNo(kwargs['ps_id'],kwargs['trial_no'],\
            kwargs['session'], kwargs['condition'])]=kwargs
        self.usedTestData[self.GetInstNo(kwargs['ps_id'],kwargs['trial_no'],\
            kwargs['session'], kwargs['condition'])]=kwargs

    def AddPs(self, ps_id, trial_no, session, condition, length, actualCat, \
            idealCat, responseCat, modelledCat=0):
        """ Here we add in a datapoint for a participant. Can also use that for
        updating the Ps's data. """
        self.trainingData[self.GetInstNo(ps_id,trial_no,session,condition)] = \
                (ps_id, trial_no, session, condition, length, actualCat, \
                idealCat, responseCat, modelledCat)
        self.usedTrainingData[self.GetInstNo(ps_id,trial_no,session,\
                condition)] = \
                (ps_id, trial_no, session, condition, length, actualCat, \
                idealCat, responseCat, modelledCat)

    def ReprPs(self, inst):
        """ Return the string representation of an instance with the following
        instance Id.
        """
        return str(inst['ps_id'])+'\t'+\
                str(inst['trial_no'])+'\t'+\
                str(inst['session'])+'\t'+\
                str(inst['condition'])+'\t'+\
                str(inst['length'])+'\t'+\
                str(inst['actualCat'])+'\t'+\
                str(inst['idealCat'])+'\t'+\
                str(inst['responseCat'])+'\t'+\
                str(inst['modelledCat'])+'\n'

    def GetInstancesForPs(self, sourceDataset='training' ,**kwargs):
        """Returns all instance ids for which the kwargs are matching with the
        instance."""
        result=[]
        if sourceDataset=='training':
            dataset=self.trainingData
        else:
            dataset=self.testData
        instances = dataset.keys()
        for inst_id in instances:
            try:
                if all([dataset[inst_id][key]==kwargs[key]\
                        for key in kwargs.keys()]):
                    result.append(inst_id)
            except KeyError, e:
                if self.verbose > 0:
                    print 'Error: keyword '+ e.message+' does not exist.'
        return result

    # DATA OUTPUT:
    def GetPsData(self,inst_no, sourceDataset='training'):
        """Return the data for this trial"""
        if sourceDataset=='training':
            dataset=self.usedTrainingData
        else:
            dataset=self.usedTestData
        try:
            return dataset[inst_no]
        except:
            if self.verbose > 0:
                print "No data for trial %d !" % inst_no
            return {'ps_id': 1, 'trial_no': 1, 'session': 1, 'condition': 1,\
                    'length': 1, 'actualCat':-1, 'idealCat': -1, 'responseCat': -1}

    def WriteOut(self,fname, pickle=False):
        """Writes out the data to the file fname. It used Python's Pickle if
        pickle==True.
        """
        if pickle:
            import pickle
        with open(fname,'w') as f:
            if self.verbose > 10:
                print "Saving data file %s" % fname
                t=time.clock()
                sys.stdout.write("[%s]" % (" " * 20))
                sys.stdout.flush()
                sys.stdout.write("\b" * (20+1))
            if pickle:
                pickle.dump(self, f)
            else:
                f.write(self.__repr__())
                f.write('ps_id,trial_no,session,condition,'+\
                        'length,actualCat,idealCat,responseCat,'+\
                        'modelledCat\n')
                lines=''
                for (i, key) in enumerate(self.trainingInstances):
                    if self.verbose > 10:
                        if (i % (len(self.trainingInstances)/20)) == 0:
                            sys.stdout.write("-")
                            sys.stdout.flush()
                    inst = self.GetPsData(key).copy()
                    try:
                        inst['modelledCat']
                    except:
                        inst['modelledCat']=self.PredictCategory(key)
                    f.write(self.ReprPs(inst))
            if self.verbose > 10:
                sys.stdout.write("\n")
                t=time.clock()-t
                print 'Time elapsed: %.2f seconds' % (t,)

    def GraphCategories(self):
        """ Makes a graph of category distribution.
        """
        # FIXME: For usedTrainingData and usedTestData
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
        As = [self.GetPsData(i)['length'] for i in self.catA if i in \
                self.trainingInstances]
        Bs = [self.GetPsData(i)['length'] for i in self.catB if i in \
                self.trainingInstances]
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
        self.forget_rate=forget_rate
        self.choice_parameter=choice_parameter
        self.noise_mu=noise_mu
        self.noise_sigma=noise_sigma

    # FORGETTING
    def ForgetOnce(self):
        """Selects instances which are to be forgotten, based on  recency of
        presentation. The probability of an item being forgotten in the end
        will be ~ exponential with ordr of presentation."""
        for (i, instNo) in enumerate(self.presentedOrder):
            if random() < self.forget_rate:
                self.ReEstimateCategory(instNo)
                self.presentedOrder.remove(instNo)
                self.presentedOrder.append(instNo) # saves the presentation order

    def PresentLoop(self):
        """ Loops through the instances in the list and after presenting every
        instance goes through the cycle of forgetting.
        If the list is None, then defaults to presenting all instances."""
        if self.verbose > 10:
            print "Presenting loop..."
            t=time.clock()
            sys.stdout.write("[%s]" % (" " * 20))
            sys.stdout.flush()
            sys.stdout.write("\b" * (20+1))
        for (i,instId) in enumerate(self.trainingInstances):
            inst=self.usedTrainingData[instId]
            cat=inst['modelledCat']
            #try:
            #    # Say, if we are presenting the data again
            #    self.presentedOrder.remove(instId)
            #except ValueError, e:
            #    # Means we haven't presented the instance yet.
            #    pass
            self.presentedOrder.append(instId) # saves the presentation order
            if self.forget_rate!=0:
                self.ForgetOnce()
            if self.verbose > 10:
                if (i % (len(self.trainingInstances)/20)) == 0:
                    sys.stdout.write("-")
                    sys.stdout.flush()
        if self.verbose > 10:
            sys.stdout.write("\n")
            t=time.clock()-t
            print 'Time elapsed: %.2f seconds' % (t,)
        # Now calculate log-likelihood
        if self.verbose > 10:
            print "Calculating log-likelihood..."
            t=time.clock()
            sys.stdout.write("[%s]" % (" " * 20))
            sys.stdout.flush()
            sys.stdout.write("\b" * (20+1))
        for (i, instId) in enumerate(self.trainingInstances):
            self.trainingData[instId]['modelledCat']=self.PredictCategory(instId, True)
            if self.verbose > 10:
                if (i % (len(self.trainingInstances)/20)) == 0:
                    sys.stdout.write("-")
                    sys.stdout.flush()
        if self.verbose > 10:
            sys.stdout.write("\n")
            t=time.clock()-t
            print 'Time elapsed: %.2f seconds' % (t,)

    # PERCEPTUAL NOISE
    def AddNoise(self, length):
        """
        Add perceptual noise.
        Controls for the case when perceptual noise is a negative number and we
        get <0 (returns 0 in such case).
        """
        return max(0,length+gauss(self.noise_mu, self.noise_sigma))

    # GCM
    def ReEstimateCategory(self,instance_no):
        """Re-estimate the category membership of the instance."""
        inst=self.usedTrainingData[instance_no]
        cat_old=inst['modelledCat'] # It defaults to actualCat in the beginning
        # anyway.
        newModelledCat = self.PredictCategory(instance_no)
        self.reEstimated += 1 # Add one to the number of re-estimated
        # instances
        if newModelledCat!=cat_old:
            self.changedCats +=1 # Add one instance to the
            # number of instances which changed category
            try:
                self.catA.remove(instance_no)
            except KeyError:
                try:
                    self.catB.remove(instance_no)
                except KeyError:
                    pass
            if newModelledCat == -1:
                self.catA.add(instance_no)
            else:
                self.catB.add(instance_no)
            self.usedTrainingData[instance_no]['modelledCat']=newModelledCat

    def ReEstimateLength(self, instance_no):
        """Re-estimate the length of an instance based on category
        membership.
        The re-estimated length is the average length of stimuli in the
        category."""

        pass

    def PredictCategory(self, instanceId, calculateLikelihood=False):
        """Return the category membership of the instance.
        Assume categories are weighted equally. The gamma parameter is taken
        from the model parameter initialisation.
        Category A == -1
        Category B == 1"""
        if (not self.catA) or (not self.catB):
            if self.verbose>100:
                print "Warning, trying to predict category on an "+\
                        "un-initialised model. Defaulting to a random "+\
                        "category."
            if random() < 0.5:
                return -1
            else:
                return 1
        else:
            psData=self.usedTrainingData[instanceId]
            length=psData['length']
            cat=psData['modelledCat']
            presented = self.Similarities[instanceId,self.presentedOrder] #select the
            # instances which were already presented
            catA_lens = presented[list(self.catA)]
            catB_lens = presented[list(self.catB)]
            sum_cat_A = np.sum(catA_lens)
            sum_cat_B = np.sum(catB_lens)
            if cat == -1:
                sum_cat_A -= 1.0
            elif cat == 1:
                sum_cat_B -= 1.0
            # We subtract 1.0, because we are also adding the similarity to
            # itself, and exp(0) = 1.
            prob_a=sum_cat_A**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
            prob_b=sum_cat_B**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
            if prob_a>=prob_b:
                if calculateLikelihood:
                    self.logLikelihood+=math.log(prob_a)
                return -1
            else:
                if calculateLikelihood:
                    self.logLikelihood+=math.log(prob_b)
                return 1

    def PredictCategoryTest(self, testSet, trainingInstances=None,\
            calculateLikelihood=True):
        """ Predict category labels for all instances in testSet, and calculate
        the log likelihood of the test set categories.
        """
        categories = []
        likelihood = 0
        for testInst in testSet:
            if (len(self.catA)<=1 and len(self.catB)<=1):
                if self.verbose>100:
                    print "Warning, trying to predict category on an "+\
                            "un-initialised model. Defaulting to a random "+\
                            "category."
                if random() < 0.5:
                    categories.append(-1)
                else:
                    categories.append(1)
                if calculateLikelihood:
                    likelihood+=math.log(0.5)
            else:
                length=testInst['length']
                sum_cat_A=sum([self.Similarity(length, self.GetPsData(inst)['length']) for \
                    inst in \
                    set(self.catA).intersection(set(trainingInstances))])
                sum_cat_B=sum([self.Similarity(length, self.GetPsData(inst)['length']) for \
                    inst in \
                    set(self.catB).intersection(set(trainingInstances))])
                prob_a=sum_cat_A**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
                prob_b=sum_cat_B**self.gamma/(sum_cat_A**self.gamma+sum_cat_B**self.gamma)
                if prob_a>=prob_b:
                    categories.append(-1)
                else:
                    categories.append(1)
                if calculateLikelihood:
                    if testInst['responseCat']==-1:
                        likelihood+=math.log(prob_a)
                    else:
                        likelihood+=math.log(prob_b)
        return (categories, likelihood)

    def ScoreModelFit(self):
        """ This returns a score of how well the model fits the data, by
        evaluating the predictions of the model.
        """
        sameCategories=[]
        pass

    def TrainGCM(self):
        """Train the GCM incrementally using all instances."""
        # First, present the instances to the model one by one.
        # Next, estimate the model response.
        # Then forget and re-estimate
        pass

#    def __init__(self):
#        """For testing. """
#        self.SetParameters(1,0.7,1,0,0.5)
#        self.verbose=100
#        # Add some instances.
#        pass

