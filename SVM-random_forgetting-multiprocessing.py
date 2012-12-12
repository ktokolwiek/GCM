#!/usr/bin/env python

import os
import math
import numpy as np
import random
import time
import argparse
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from operator import itemgetter
from sklearn.svm import SVC
#This deosn't seem to work on mac:
#from scikits.learn.svm import SVC

#edit by Lukasz 30/10/12
#how to select the instance to be forgotten - uniformly or normally?
#according to which normal distribution?
def select_uniformly(from_list_instances, from_list_labels):
    """ Here we sample uniformly from from_list_instances, and we then
    re-sample it using the `resample` function. Then we return the new
    distribution of category labels.
    """
    #make sure the lists are equal lenghts
    assert len(from_list_instances) == len(from_list_labels)
    #get the index of the element we need to forget
    a=random.randint(0,len(from_list_instances)-1)
    #now re-sample the instance w.r.t. to the remaining instances
    re_sampled=resample(a,from_list_instances, from_list_labels,get_distances(from_list_instances),5)
    #and put it in the original labels instead of the old label
    return from_list_labels[:a]+[re_sampled]+from_list_labels[a+1:]

def categorise_ideally(instance):
    """ Potentially could put here a more complex function for detecting the
    category of an instance.
    """
    if instance<37.5:
        return -1
    else:
        return 1


#DOESN'T REALLY MAKE SENSE TO SAMPLE FROM A DIFFERENT DISTRIBUTION THAN FLAT?
# The data are distributed ~ Gaussian anyway, so why would I do that?
# 
#def update_mu_sigma(old_mu_A, old_sigma_A, N_A, old_mu_B, old_sigma_B, N_B,
#        instance, from_A_to_B):
#    """
#    Updates the standard deviations and the means of two distributions A and B.
#    from_A_to_B is True when the instance moved from category A to category B,
#    and False otherwise.
#    """
#    if from_A_to_B:
#        new_mu_A=(old_mu_A*N_A - instance)/N_A-1
#        new_N_A = N_A-1
#        new_sd_A=()
#    new_mu_A
#
#def select_normal(from_list_instances, from_list_labels):
#    """
#    This will sample an instance from from_instances_list w.r.t its mu and
#    sigma, and then forget it and resmaple using `resample`.
#    To implement: see how to calculate mean / st. dev. efficiently.
#    Can compute once outside and then update it.
#    """
#    return from_list_labels

def read_training_data(ps_numbers=['112101']):
    """ Reads the training instances for a single participant from the
    filesystem in the defined path.
    The data are strutured as a list of lists per each participant
    -participant ID
    -session 1..5
    -feedback 1 - actual 2 - idealised
    -length in mm
    -distribution/feedback category -1:short 1:long
    -participant/response category -1:short 1:long
    -distance matrix, which is NxN
    """
    PATH_TO_FILES='.'
    f_path = os.path.join(PATH_TO_FILES,'Randall_learn.txt')
    with open(f_path) as f:
        data = []
        thisPs=[[] for i in range(7)]
        lastPs=''
        for line in f:
            cols=line.split()
            if cols[0] in ps_numbers:
                if lastPs!=cols[0]:
                    # if found a new participant
                    if lastPs!='':
                        #if it is not the beginning of file then
                        # calculate the distances between training instances for
                        # this Ps:
                        thisPs[6]=(get_distances(thisPs[3]))
                        # append the Ps data to the big database
                        data.append(thisPs)
                        if DEBUG>=100:
                            print "Read Participant training data for Ps ID: %s" % thisPs[0]
                        # clear it for the next Ps
                        thisPs=[[] for i in range(7)]
                    lastPs=cols[0] #update the current Ps number
                    # add the Ps ID to the data
                    thisPs[0]=lastPs
                if cols[9]=='A':
                    idealCat=-1
                else:
                    idealCat=1
                if cols[10]=='A':
                    participantCat=-1
                else:
                    participantCat=1
                thisPs[1].append(int(cols[1])) #add session
                thisPs[2].append(int(cols[5])) #add feedback type
                thisPs[3].append(int(cols[8])) #add length
                thisPs[4].append(idealCat) #add ideal category
                thisPs[5].append(participantCat) #add response
        if lastPs!='':
            #if it is not the beginning of file then
            # calculate the distances between training instances for
            # this Ps:
            thisPs[6]=(get_distances(thisPs[3]))
            # append the Ps data to the big database
            data.append(thisPs)
            if DEBUG>=100:
                print "Read Participant training data for Ps ID: %s" % thisPs[0]
            # clear it for the next Ps
            thisPs=[[] for i in range(7)]
    return data

def read_test_data(ps_numbers=['112101']):
    """ Reads the test instances for a single participant from the
    filesystem in the defined path.
    The data are strutured as a list of lists per each participant
    -participant ID
    -session 1 to 5 - some Ps don't have all sessions' data
    -feedback 1 - actual 2 - idealised
    -length in mm
    -distribution/feedback category -1:short 1:long
    This is calculated based on line length.
    -participant/response category -1:short 1:long
    """
    PATH_TO_FILES='.'
    f_path = os.path.join(PATH_TO_FILES,'Randall_test.txt')
    with open(f_path) as f:
        data = []
        thisPs=[[] for i in range(6)]
        lastPs=''
        for line in f:
            cols=line.split()
            if cols[0] in ps_numbers:
                if lastPs!=cols[0]:
                    # if found a new participant
                    if lastPs!='':
                        #if it is not the beginning of file then
                        # append the Ps data to the big database
                        data.append(thisPs)
                        if DEBUG>=100:
                            print "Read Participant test data for Ps ID: %s" % thisPs[0]
                        # clear it for the next Ps
                        thisPs=[[] for i in range(6)]
                    lastPs=cols[0] #update the current Ps number
                    # add the Ps ID to the data
                    thisPs[0]=lastPs
                idealCat=categorise_ideally(cols[8])
                if cols[10]=='A':
                    participantCat=-1
                else:
                    participantCat=1
                thisPs[1].append(int(cols[1])) #add session 1 or 5
                thisPs[2].append(int(cols[5])) #add feedback type
                thisPs[3].append(int(cols[8])) #add length
                thisPs[4].append(idealCat) #add ideal category
                thisPs[5].append(participantCat) #add response
        if lastPs!='':
            #if it is not the beginning of file then
            # append the Ps data to the big database
            data.append(thisPs)
            if DEBUG>=100:
                print "Read Participant test data for Ps ID: %s" % thisPs[0]
            # clear it for the next Ps
            thisPs=[[] for i in range(6)]
    return data


def get_distances(instances):
    """
    Naively calculates pairwise distances between all instances and returns an
    appropriate matrix to be used with kNN.
    """
    return [[abs(frm - to) for frm in instances] for to in instances]


def resample(instance_ID, instances, labels, distances, k=5, recency = False):
    """ Re-sample the label of an instance with respect to the distribution of
    instances in instances and distribution of labels in labels
    .
    Implement here kNN, possibly using multiprocessing, and hope it works. k is
    the parameter in kNN, and distances is a 2-d list with distances from
    instance [i] to [j]
    .
    Also implemented recency - simply get k items last in memory / last
    presented.
    """
    #sanity check
    assert len(labels)==len(instances)
    if recency:
        if instance_ID<1:
            # If no recent data, then assign randomly
            cat=random.random()-0.5
        else:
            kNN=zip(instances[max(instance_ID-k,0):max(0,instance_ID-0)],\
                labels[max(instance_ID-k,0):max(0,instance_ID-0)])
    else:
        # gets tuples (distance, label), sorted by distance
        distance_label_pairs = sorted(zip(distances[instance_ID],labels))
        # top k of these:
        kNN=distance_label_pairs[:k]
        # get the arithmetic mean of labels. If>0 then mostly 1, if <=0, then mostly -1.
        # TODO maybe should do a weighted average following Poisson-like?
        cat = np.mean([neighbour[1] for neighbour in kNN])
    if cat>0:
        return 1
    else:
        return -1

def forget(ps_data, rate=0.01, k=5):
    """ Goes through one cycle of forgetting.
    Will be good to implement here a forgetting rate.
    """
    # These are just so that I remember which one is which:
    #instances = ps_data[3]
    #labels=ps_data[4] (ideal)
    #distances=ps_data[6]
    for (inst_ID,instance) in enumerate(ps_data[2]):
        if random.random()<rate: #forgetting rate
            ps_data[4][inst_ID]=resample(inst_ID,ps_data[3],ps_data[4],ps_data[6],k)
            #here the labels are updated instantly, should they?
    return ps_data[4] #only need to return the labels, nothing else changes

def forget_loop(ps_data, rate=0.1, k=5, N=50):
    """ Loops through participants' data and applies kNN to each instance.
    """
    if DEBUG>=1000:
        print "Re-sampling participant ID %s with forgetting rate= %s and k= %s"\
                %(ps_data[0],rate,k)
    for i in range(N):
        ps_data[4]=forget(ps_data, rate, k)
    return ps_data


def fit_SVM((train_data, test_data)):
    """ The SVM is fitted with various gamma and cost. Then the test data are
    predicted using the models and rated on some information theory measure.
    """
    assert train_data[0]==test_data[0]
    #We are looking at the same Ps ID.
    instances_train=np.vstack(train_data[3])
    labels_train=np.hstack(train_data[4])
    # These two are used in training the model
    instances_test=np.vstack(test_data[3])
    labels_test=np.hstack(test_data[5]) # this is the participant's response
    # These two are used in model evauation
    gamma_values = np.arange(.01, .06, .01)
    k_array = np.arange(1,6,1) #Cost element
    maxScoreTest = -1 #arbitrarily low magic number
    fit_k,fit_gamma = -1,-1 #for debugging
    bestSVM =[]
    for gam in gamma_values:
        if DEBUG>=100:
            print "Fitting SVM for Ps %s with gamma %s"%(train_data[0], gam)
        clf = SVC(kernel='rbf',gamma=gam)
        # Multidimensional Gaussian kernel
        clf.fit(instances_train, labels_train)
        # Simple.
        #
        support_vector_positions = clf.support_
        #print support_vector_positions, len(support_vector_positions)
        support_vector_list = clf.support_vectors_
        #print support_vector_list, len(support_vector_list)
        label_weight_list = clf.dual_coef_
        #print label_weight_list, len(label_weight_list[0])
        bias = clf.intercept_[0]
        # BIAS is here - the intercept of the SVM
        for k in k_array:
            scoreTest = scoring_predictions_SVM(instances_test, labels_test, gam, support_vector_list, label_weight_list, k, bias)
            if scoreTest>maxScoreTest:
                fit_k = k
                fit_gamma = gam
                maxScoreTest = scoreTest
                bestSVM=clf
                if DEBUG>=1000:
                    print "Ps: %s ScoreTest: %s, k: %s, gamma: %s"%\
                            (train_data[0],scoreTest,fit_k,fit_gamma)
    if WRITE_TO_DISK:
        #get the path
        if DEBUG>100:
            print "Writing to disk. Predictions for Ps %s."%test_data[0]
        path=os.path.join('.',PATH_OUT,'BestSVMFit-p'+test_data[0]+'-session1.txt')
        matching = 0.0
        N=0.0
        lines = []
        with open(path,'w') as f:
            f.write("Ps: %s ScoreTest: %s, k: %s, gamma: %s\n"%\
                (test_data[0],scoreTest,fit_k,fit_gamma))
            f.write('Ps\tSession\tFeedback\tLength\tIdeal\tResp\tSVM_resampling_prediction\t'+\
                    'Correct_with_resampling\tSVM_no_resampling_prediction\tCorrect_no_resampling\t'+\
                    'Fitted_k\tFitted_gamma\tFitted_kNN\tFitted_forgetting_rate\tPerc_correct_with'+\
                    '\tPerc_correct_no\n')
            for i in range(len(test_data[3])):
                predict = bestSVM.predict(test_data[3][i])[0]
                correct = 'No'
                N+=1
                if predict==test_data[5][i]:
                    correct='Yes'
                    matching+=1
                lines.append("%(ps)s\t%(session)s\t%(feedback)s\t%(length)s\t%(ideal)s\t%(resp)s\t%(svm_resampling)s\t%(correct_resampling)s\t%(svm_no)s\t%(correct_no)s\t%(fit_k)s\t%(fit_gamma)s\t%(fit_kNN)s\t%(fit_rate)s\t"% \
                        {'ps': test_data[0],\
                        'session': test_data[1][i],\
                        'feedback': test_data[2][i],\
                        'length': test_data[3][i],\
                        'ideal': test_data[4][i],\
                        'resp': test_data[5][i],\
                        'svm_resampling': '',\
                        'correct_resampling': '',\
                        'svm_no': predict,\
                        'correct_no': correct,\
                        'fit_k': fit_k,\
                        'fit_gamma': fit_gamma,\
                        'fit_kNN': '',\
                        'fit_rate': ''})
            for line in lines:
                # write out the percent correct matches in every line in the
                # file.
                f.write(line+'\t'+str(matching/N)+'\n')
    return (fit_k, fit_gamma)

def fit_SVM_resample((train_data, test_data, (k, gamma))):
    """ This tries out some parameters of the resampling: forgetting rate and k
    (not to be mistaken with the k from SVM fit)
    """
    assert train_data[0]==test_data[0] #we use the same Ps data
    assert ((test_data[1][0]==1) or (test_data[1][0]==5)) #make sure the right training session is used
    maxScoreTest = -1 #arbitrarily low magic number
    fit_rate, fit_kNN = -1,-1
    best_resample = train_data
    bestSVM=[]
    for kNN in [3,5,7,9,13,17,20]:#8 of these
        for forget_rate in [0.0001,0.001,0.005,0.01,0.05,0.1]:#,0.2,0.3,0.4,0.5]:#10 of these
            try:
                instances_train=np.vstack(train_data[3])
                labels_resampled = forget_loop(train_data,forget_rate,kNN)[4]
                labels_train=np.hstack(labels_resampled)
                #USE THE RESAMPLED LABELS HERE - only the labels ([4])
                # These two are used in training the model
                instances_test=np.vstack(test_data[3])
                labels_test=np.hstack(test_data[5]) # this is the participant's response
                # These two are used in evaluation
                #
                #TRAIN THE MODEL HERE:
                #
                clf = SVC(kernel='rbf',gamma=gamma)
                # Multidimensional Gaussian kernel
                clf.fit(instances_train, labels_train)
                # Simple.
                #
                support_vector_positions = clf.support_
                support_vector_list = clf.support_vectors_
                label_weight_list = clf.dual_coef_
                bias = clf.intercept_[0]
                # BIAS is here - the intercept of the SVM
                scoreTest = scoring_predictions_SVM(instances_test, labels_test, \
                        gamma, support_vector_list, label_weight_list, k, bias)
                if scoreTest>maxScoreTest:
                    fit_kNN = kNN
                    fit_rate = forget_rate
                    bestSVM=clf
                    maxScoreTest = scoreTest
                    if DEBUG>=1000:
                        print "Ps: %s ScoreTest: %s, kNN: %s, forgetting rate: %s"% \
                                (test_data[0],scoreTest,fit_kNN,fit_rate)
            except:
                if DEBUG>=100:
                    print "Skipped kNN %s and forget rate %s because all labels were the same." %\
                            (kNN,forget_rate)
    if WRITE_TO_DISK:
        # here you have two SVMs - one trained on data which was not
        # re-sampled, and one which was trained on the best-fitting re-sampled
        # data
        path=os.path.join('.',PATH_OUT,'BestSVMFit-p'+test_data[0]+'-session5.txt')
        # That is for tracking prediction accuracy
        matching_with_resampling = 0.0
        matching_no_resampling = 0.0
        N=0.0
        instances_train=np.vstack(train_data[3])
        labels_train=np.hstack(train_data[4])
        SVM_no_resampling = SVC(kernel='rbf',gamma=gamma)
        SVM_no_resampling.fit(instances_train,labels_train)
        lines = []
        with open(path,'w') as f:
            f.write('Ps: %s ScoreTest: %s, k: %s, gamma: %s, kNN: %s, forgetting rate: %s\n'% \
                            (test_data[0],scoreTest,k,gamma,fit_kNN,fit_rate))
            f.write('Ps\tSession\tFeedback\tLength\tIdeal\tResp\tSVM_resampling_prediction\t'+\
                    'Correct_with_resampling\tSVM_no_resampling_prediction\tCorrect_no_resampling\t'+\
                    'Fitted_k\tFitted_gamma\tFitted_kNN\tFitted_forgetting_rate\tPerc_correct_with'+\
                    '\tPerc_correct_no\n')
            for i in range(len(test_data[1])):
                predict_no_resampling = \
                        SVM_no_resampling.predict(test_data[3][i])[0]
                predict_with_resampling = bestSVM.predict(test_data[3][i])[0]
                correct_no_resampling = 'No'
                correct_with_resampling = 'No'
                N+=1
                if predict_with_resampling ==test_data[5][i]:
                    correct_with_resampling='Yes'
                    matching_with_resampling+=1
                if predict_no_resampling == test_data[5][i]:
                    correct_no_resampling = 'Yes'
                    matching_no_resampling += 1
                lines.append("%(Ps)s\t%(Session)s\t%(Feedback)s\t%(Length)s\t%(Ideal)s\t%(Resp)s\t%(SVM_resampling)s\t%(Correct_resampling)s\t%(SVM_no)s\t%(Correct_no)s\t%(fit_k)s\t%(fit_gamma)s\t%(fit_kNN)s\t%(fit_rate)s\t"% \
                        {'Ps': test_data[0],\
                        'Session': test_data[1][i],\
                        'Feedback': test_data[2][i],\
                        'Length': test_data[3][i],\
                        'Ideal': test_data[4][i],\
                        'Resp': test_data[5][i],\
                        'SVM_resampling': predict_with_resampling,\
                        'Correct_resampling': correct_with_resampling,\
                        'SVM_no': predict_no_resampling,\
                        'Correct_no': correct_no_resampling,\
                        'fit_k': k,\
                        'fit_gamma': gamma,\
                        'fit_kNN': fit_kNN,\
                        'fit_rate': fit_rate})
            for line in lines:
                f.write(line+str(matching_with_resampling/N)+'\t'+str(matching_no_resampling/N)+'\n')
    return (fit_rate, fit_kNN)

def run_experiment():
    IDs=['112101', '112102', '112103', '112104', '112105', '112106', '112107',\
            '112108', '112109', '112110', '112202', '112203',\
            '112204', '112205', '112206', '112207', '112208', '112209',\
            '112210', '112211', '122101', '122102', '122103', '122104',\
            '122105', '122106', '122107', '122108', '122110',\
            '122201', '122202', '122203', '122204', '122205', '122206',\
            '122207', '122208', '122209', '122210']
    # '112201', '122109' does not have data for day 5. This took me the whole weekend to
    # figure out.
    #IDs=IDs[24:] #for testing so that it finishes quickly
    if DEBUG>=10:
        print "\nReading data...\n"
    train = read_training_data(IDs)
    test = read_test_data(IDs)
    pool = Pool(cpu_count())
    test_data_session_1 = []
    test_data_session_5 = []
    for t in test:
        sess_1 = [t[0],[],[],[],[],[]]
        sess_5 = [t[0],[],[],[],[],[]]
        for i in range(len(t[1])):
            if t[1][i]==1:
                #if the datum comes from the first session
                for j in range(1,6):
                    sess_1[j].append(t[j][i])
            else:
                for j in range(1,6):
                    sess_5[j].append(t[j][i])
        test_data_session_1.append(sess_1)
        test_data_session_5.append(sess_5)
    SVM_data = [(train[i],test_data_session_1[i]) for i in range(len(train))]
    if DEBUG>=10:
        print "\nFitting SVM to first training session...\n"
    SVM_fits = pool.map(fit_SVM,SVM_data)
    SVM_data_resampled = [(train[i], test_data_session_5[i],\
                SVM_fits[i]) for i in range(len(train))]
    if DEBUG>=10:
        print "\nFitting SVM to last training session...\n"
    resampled_fits = pool.map(fit_SVM_resample, SVM_data_resampled)
    time.sleep(1)


#avoiding negative numbers due to machine precision
def adjusting(number):
    if number<=0.0:
        return 0.0000000000001
    else:
        return number

#determining potential score according to parameters    
def scoring_predictions_SVM(instances, labels, gamma, support_vector_list, label_weight_list, k, bias):
    """ Returns the average score for the fit between insances in X_test_list
    and the SVM vectors-defined partition.
    instances: an iterable containing test instances
    labels: an iterable containing test set labels
    gamma: The gamma coefficient in the kernel function
    support_vector_list: the SVM - list of salient instances, exemplars or so.
    label_weight_list: ?
    k: cost ?
    bias: Bias towards one category, added to every similarity score of every
    instance
    """
    score=0.0
    for ID, instance in enumerate(instances):
        simA, simB = 0.0, 0.0
        if bias<0.0:
            simA = abs(bias)
        elif bias>0.0:
            simB = abs(bias)
        for item in range(len(support_vector_list)):
            sim = similarity_function(instance[0], support_vector_list[item], gamma)
            label_weight = label_weight_list[0][item]
            add = -label_weight * sim
            if add<0.0:
                simA += abs(add)
            elif add>0.0:
                simB += abs(add)
        probA = 1/(1+((simB/simA)**k)) #why this?
        if labels[ID]==-1:
            score_item = probA
        else:
            score_item = 1-probA
        score += score_item
    return score/len(instances)

#computing similarity between games
def similarity_function(test_item, support_vector, gamma):
    return math.exp(-gamma*(((test_item-support_vector)**2))) #Why use abs()?
# That uses a different gamma in every call

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fit SVMs using stochastic '+\
            're-sampling')
    parser.add_argument('-p', '--path', dest='path', type=str, nargs='?',\
            default='results', help="The model's output path. Created if necessary")
    parser.add_argument('-d', '--debug', dest="debug", type=int, nargs='?',\
            default='1000', help="Verbosity level")
    return parser.parse_args()

#Do some arguments-reading
args=parse_arguments()
PATH_OUT=args.path
try:
    os.makedirs(PATH_OUT)
except Exception as e:
    print e
DEBUG=args.debug #set to 0 for non-verbose execution
WRITE_TO_DISK=True

run_experiment()
