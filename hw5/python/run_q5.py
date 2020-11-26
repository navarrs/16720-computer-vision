from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt
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
learning_rate = 3e-5
hidden_size = 32
in_size = 1024
out_size = 1024
lr_rate = 20
batches = get_random_batches(
    train_x, np.ones((train_x.shape[0], 1)), batch_size)
batch_num = len(batches)
params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(in_size=in_size, out_size=hidden_size,
                   params=params, name='elyr1')
initialize_weights(in_size=hidden_size, out_size=hidden_size,
                   params=params, name='eout')
initialize_weights(in_size=hidden_size, out_size=hidden_size,
                   params=params, name='dlyr1')
initialize_weights(in_size=hidden_size, out_size=out_size,
                   params=params, name='dout')


def update(params, name, lr, mu=0.9):
    params['m_W' + name] = mu * params['m_W' + name] - \
        lr * params['grad_W' + name]
    params['W' + name] = params['W' + name] + params['m_W' + name]
    params['m_b' + name] = mu * params['m_b' + name] - \
        lr * params['grad_b' + name]
    params['b' + name] = params['b' + name] + params['m_b' + name]


# should look like your previous training loops
loss = []
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2 = x**2 - 2xy + y**2 => 2x -2y
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        enc_h1 = forward(xb, params, 'elyr1', activation=relu)
        enc_out = forward(enc_h1, params, 'eout', activation=relu)
        dec_h1 = forward(enc_out, params, 'dlyr1', activation=relu)
        dec_out = forward(dec_h1, params, 'dout', activation=sigmoid)

        total_loss += np.sum((xb - dec_out)**2) / (batch_num * batch_size)

        delta1 = 2.0 * (dec_out - xb)
        delta2 = backwards(delta1, params, 'dout',
                           activation_deriv=sigmoid_deriv)
        delta3 = backwards(delta2, params, 'dlyr1',
                           activation_deriv=relu_deriv)
        delta4 = backwards(delta3, params, 'eout', activation_deriv=relu_deriv)
        backwards(delta4, params, 'elyr1', activation_deriv=relu_deriv)

        update(params, 'elyr1', learning_rate)
        update(params, 'eout', learning_rate)
        update(params, 'dlyr1', learning_rate)
        update(params, 'dout', learning_rate)
        
    loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d}\tloss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9


# Q5.3.1
epochs = np.arange(start=0, stop=max_iters, step=1)
plt.plot(epochs, loss)
plt.title('Autoencoder Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.savefig("../out/q5/autoencoder_loss.png")
plt.show()
plt.close()
# visualize some results
##########################
##### your code here #####
##########################
valid_y = valid_data['valid_labels']
labels = np.random.choice(35, size=(5), replace=False)
valid_labels = np.argmax(valid_y, axis=1)

for label in labels:
    idx = np.random.choice(np.where(valid_labels == label)[0], size=(2))
    for i in idx:
        eh1 = forward(valid_x[i], params, 'elyr1', activation=relu)
        eout = forward(eh1, params, 'eout', activation=relu)
        dh1 = forward(eout, params, 'dlyr1', activation=relu)
        y = forward(dh1, params, 'dout', activation=sigmoid).reshape(32, 32).T

        plt.subplot(1, 2, 1)
        plt.imshow(valid_x[i].reshape(32, 32).T)
        plt.title('Ground Truth')
        plt.subplot(1, 2, 2)
        plt.imshow(y)
        plt.title('Generated')
        # plt.savefig(f"../out/q5/label-{label}_idx-{i}_loss-{total_loss}.png", 
        #             bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()


# Q5.3.2
# evaluate PSNR
##########################
##### your code here #####
##########################
PSNR = 0.0
N = len(valid_x)
for vx in valid_x:
    eh1 = forward(vx, params, 'elyr1', activation=relu)
    eout = forward(eh1, params, 'eout', activation=relu)
    dh1 = forward(eout, params, 'dlyr1', activation=relu)
    y = forward(dh1, params, 'dout', activation=sigmoid)

    PSNR += psnr(vx, y)
print(f"PSNR for the {N} val images is: {PSNR/N}")
