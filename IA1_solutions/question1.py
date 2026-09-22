'''Splice junctions are points on a DNA sequence at which 'superfluous' DNA is removed during
the process of protein creation in higher organisms. The problem posed in this dataset is to recognize,
given a sequence of DNA, the boundaries between exons (the parts of the DNA sequence retained after splicing)
and introns (the parts of the DNA sequence that are spliced out). This problem consists of two subtasks:
recognizing exon/intron boundaries (referred to as El sites), and recognizing intron/exon boundaries
(IE sites). (In the biological community, IE borders are referred to as "acceptors" while El borders are
referred to as "donors".)
'''
# Using the UCI Splice-junction Gene Sequences dataset, build and evaluate a FFN deep-learning model that
# classify each 60-base DNA sequence as:

# - El: exon-intron boundary
# - IE: intron-exon boundary
# - N: neither

'''The dataset contains 3,190 labelled sequences. Do not use the instance-name field as an input feature.
You can use one-hot representation to encode a base. Use train-test-split (80:20) with stratification
to form train and test sets. Use the seed value of 42 for train-test-split. You need to construct a 
feed forward deep neural network of your choice to maximize the performance. However, it is essential 
that there are at least two FC layers. Print the training loss at the end of each epoch. You need to
train for atleast 5 epochs. Print the overall accuracy and the macro F1-score on the test set using the
trained model.'''

# Tasks:
#
#     Loading the dataset (5 Marks)
#
#     Preprocess the sequences (10 Marks)
#
#     Build a deep neural network & report the results (20 Marks)
