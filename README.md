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

# 6 February 2013
# Should maybe add an extra argument to the functions, being a list of
# instances which are used for, say, evaluation of the model, or prediction of
# categories.
# Also - WriteOut() is terribly slow. Maybe just use pickle?

# 7 February
# Added log-likelihood estimation and a simple scratch of a test case in
# testcase.py. Also, everything works faster becasue of pre-computing a set of
# instances to be considered as opposed to matching every instance in the list
# of instances.

# 8 February
# Added input of test data, and will be evaluating the model on test data. Also
# analogous methods for obtaining an instance, updating etc. from the test
# dataset.
# I will now need to see how to evaluate the model before the forgetting loop.

# 18 February
# Had a break to program the experiment (canela.py). This is on the love01
# server.
# Now I am trying to optimise the code through profiling. I will need a way to
# efficiently compute similarity measures, because that is what takes up most of
# the time now, together with the overhead of PredictCategory.
# A very simple solution would be to pre-compute the similarity measure and not
# use noise so that we can address it by a small finite numebr of integers.

# 19 Feb
# Messing around with ndarrays
