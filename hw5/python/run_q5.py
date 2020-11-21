import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
in_size = 1024
out_size =1024
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(in_size=in_size, out_size=hidden_size, 
                   params=params, name='enc_l1')
initialize_weights(in_size=hidden_size, out_size=hidden_size, 
                   params=params, name='enc_out')
initialize_weights(in_size=hidden_size, out_size=hidden_size, 
                   params=params, name='dec_l1')
initialize_weights(in_size=hidden_size, out_size=out_size, 
                   params=params, name='dec_out')


# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        enc_h1 = forward(xb, params, 'enc_l1', activation=relu)
        enc_out = forward(enc_h1, params, 'enc_out', activation=relu)
        dec_h1 = forward(enc_out, params, 'dec_l1', activation=relu)
        dec_out = forward(dec_h1, params, 'dec_out', activation=sigmoid)
        
        loss = (xb-dec_out)**2
        
        delta1 = 2*xb - 2*dec_out

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
