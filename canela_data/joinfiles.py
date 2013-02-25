#!/usr/bin/python

import os

files = [f for f in os.listdir('.') if f.endswith('.txt')]

fname_training = "canela_training.txt"
fname_test = "canela_test.txt"

training_data = []
test_data = []

with open(files[0]) as f:
    header = f.readline()
    training_data.append(header)
    test_data.append(header)

for fname in files:
    with open(fname) as f:
        data = f.readlines()[1:]
        training_data.extend(data[:160])
        test_data.extend(data[160:])

with open(fname_training, 'w') as f_train:
    f_train.writelines(training_data)

with open(fname_test, 'w') as f_test:
    f_test.writelines(test_data)

