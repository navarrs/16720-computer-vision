import string
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 30
# pick a batch size, learning rate
batch_size = 72
learning_rate = 1e-3  # best: 0.01
hidden_size = 64
##########################
##### your code here #####
##########################
n_cls = len(train_y[0])
in_size = train_x.shape[1]
batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)
print(f"batches: {batch_num}")

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(in_size, hidden_size, params, 'layer1')
initialize_weights(hidden_size, n_cls, params, 'output')

init_w = copy.deepcopy(params['Wlayer1'])

# with default settings, you should get loss < 150 and accuracy > 80%
train_losses, train_accs = [], []
valid_losses, valid_accs = [], []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        # forward
        h1 = forward(xb, params, 'layer1', activation=sigmoid)
        yp = forward(h1, params, 'output', activation=softmax)

        # loss and accuracy
        loss, acc = compute_loss_and_acc(yb, yp)
        total_loss += loss / (batch_num * len(xb))
        total_acc += acc / batch_num

        # backward
        delta1 = yp - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        # xb = xb - learning_rate * grad_x
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']

    # Compute validation loss and accuracy so far
    vh1 = forward(valid_x, params, 'layer1', activation=sigmoid)
    vyp = forward(vh1, params, 'output', activation=softmax)
    loss, acc = compute_loss_and_acc(valid_y, vyp)
    loss /= len(valid_x)
    valid_losses.append(loss)
    valid_accs.append(acc)

    # Append running loss and accuracy
    train_losses.append(total_loss)
    train_accs.append(total_acc)

    if itr % 2 == 0:
        print("itr {:02d}".format(itr))
        print("\t train:\t loss: {:.3f}\t acc: {:.3f}"
              .format(total_loss, total_acc))
        print("\t valid:\t loss: {:.3f}\t acc: {:.3f}"
              .format(loss, acc))
print("itr {:02d}".format(max_iters))
print("\ttrain:\t loss: {:.3f}\t acc: {:.3f}".format(total_loss, total_acc))
print("\tvalid:\t loss: {:.3f}\t acc: {:.3f}".format(loss, acc))

# Plot loss and accuracy
epochs = np.arange(start=0, stop=max_iters, step=1)

fig, axs = plt.subplots(1, 2, constrained_layout=True)
fig.suptitle(
    f"iters:{max_iters} - batch-size: {batch_size} - lr: {learning_rate}",
    fontsize=12)
axs[0].plot(epochs, train_losses, 'k.-', label='train loss')
axs[0].plot(epochs, valid_losses, 'g.-', label='val loss')
axs[0].set_title('Loss vs Epochs')
axs[0].set_xlabel('epochs ')
axs[0].set_ylabel('loss')

axs[1].plot(epochs, train_accs, 'k.-', label='train accuracy')
axs[1].plot(epochs, valid_accs, 'g.-', label='valid accuracy')
axs[1].set_title('Accuracy vs Epochs')
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('accuracy')
plt.legend(loc='upper left', borderaxespad=0.)

plt.savefig("../out/q3/loss-{:.3f}_acc-{:.3f}_iter-{}_lr-{}_batch-{}.png"
            .format(total_loss, total_acc, max_iters, learning_rate, batch_size))

plt.show()

# run on validation set and report accuracy! should be above 75%
##########################
##### your code here #####
##########################
h1 = forward(valid_x, params, 'layer1', activation=sigmoid)
yp = forward(h1, params, 'output', activation=softmax)
_, valid_acc = compute_loss_and_acc(valid_y, yp)
print(f'Validation accuracy: {valid_acc}')
if False:  # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32, 32).T)
        plt.show()
saved_params = {k: v for k, v in params.items() if '_' not in k}
with open(f'q3_weights_{loss}.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3

# visualize weights here
##########################
##### your code here #####
##########################
F = plt.figure(1, (32, 32))
grid = ImageGrid(F, 111, nrows_ncols=(8, 8), axes_pad=0.05)

learned_w = params['Wlayer1']
for i in range(hidden_size):
    # w = init_w[:, i].reshape(32, 32)
    w = learned_w[:, i].reshape(32, 32)

    ax = grid[i]
    ax.imshow(w, origin="lower", interpolation="nearest")


plt.draw()
# plt.savefig("../out/q3/init_weights.png")
# plt.savefig("../out/q3/weights_loss-{:.3f}_acc-{:.3f}_iter-{}_lr-{}_batch-{}.png"
#             .format(total_loss, total_acc, max_iters, learning_rate, batch_size))

plt.show()
# plt.close()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))

# compute confusion matrix here
##########################
##### your code here #####
##########################
yl = np.argmax(valid_y, axis=1)
yh = np.argmax(yp, axis=1)
for i in range(yl.shape[0]):
    confusion_matrix[yl[i], yh[i]] += 1

plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] +
           ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] +
           ''.join([str(_) for _ in range(10)]))
# plt.savefig("../out/q3/confmat_loss-{:.3f}_acc-{:.3f}_iter-{}_lr-{}_batch-{}.png"
#             .format(total_loss, total_acc, max_iters, learning_rate, batch_size))
plt.show()
# plt.close()
