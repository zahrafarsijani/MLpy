## ------
## Nerual Network implemented using Tensorflow



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.contrib.eager as tfe


# read the training set
data_train = np.array(pd.read_hdf("train.h5","train"))
X = data_train[:, 1:]
y = data_train[:, 0]

# read the test set
X_test = np.array(pd.read_hdf("test.h5","test"))

# Standardize and Normalize the train and test data
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)


#print (np.shape(X))
#print (np.shape(y))
#print (np.shape(X_test))

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.1)

# Set Eager API
tfe.enable_eager_execution()

# Neural Net implementation

# Parameters (hyper parameters)
file_name = 'predictions.csv'
learning_rate = 0.001
num_steps = 1000
batch_size = 5000
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer; num of neurons
n_hidden_2 = 256 # 2nd layer; num of neurons 
n_hidden_3 = 256
n_hidden_4 = 256
n_hidden_5 = 256
n_hidden_6 = 32
n_hidden_7 = 0
n_hidden_8 = 0
n_hidden_9 = 0

# in case MNIST is used
num_input = 100 # MNIST data input (img shape: 28*28)
num_classes = 5 # MNIST total classes (0-9 digits)

# Use TF Dataset to split data into batches
dataset = tf.data.Dataset.from_tensor_slices(
    (trainX,trainY)).batch(batch_size)
dataset_iter = tfe.Iterator(dataset)


# Define a neural network.
# Here I instantiate a tfe.Network class (to use eager API and tf.layers)
class NeuralNet(tfe.Network):
    def __init__(self):
        # Define each layer
        super(NeuralNet, self).__init__()

        # Hidden fully connected layer with 256 neurons
        self.layer1 = self.track_layer(
            tf.layers.Dense(n_hidden_1, activation=tf.nn.relu))

        # Hidden fully connected layer with 256 neurons
        self.layer2 = self.track_layer(
            tf.layers.Dense(n_hidden_2, activation=tf.nn.relu))

        # Hidden fully connected layer with 256 neurons
        self.layer3 = self.track_layer(
            tf.layers.Dense(n_hidden_3, activation=tf.nn.relu))

        # Hidden fully connected layer with 256 neurons
        self.layer4 = self.track_layer(
            tf.layers.Dense(n_hidden_4, activation=tf.nn.relu))

        # Hidden fully connected layer with 256 neurons
        self.layer5 = self.track_layer(
            tf.layers.Dense(n_hidden_5, activation=tf.nn.relu))

        # Hidden fully connected layer with 256 neurons
        self.layer6 = self.track_layer(
            tf.layers.Dense(n_hidden_6, activation=tf.nn.relu))

        # Hidden fully connected layer with 256 neurons
        # self.layer7 = self.track_layer(
        #     tf.layers.Dense(n_hidden_7, activation=tf.nn.relu))

        # self.layer8 = self.track_layer(
        #     tf.layers.Dense(n_hidden_8, activation=tf.nn.relu))
        #
        # self.layer9 = self.track_layer(
        #     tf.layers.Dense(n_hidden_9, activation=tf.nn.relu))

        # Output fully connected layer with a neuron for each class
        self.out_layer = self.track_layer(tf.layers.Dense(num_classes))

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # x = self.layer7(x)  # to have more layers, uncommnet the lines accordingly
        # x = self.layer8(x)
        # x = self.layer9(x)
        return self.out_layer(x)

#instantiate a nn object 
neural_net = NeuralNet()


# Cross-Entropy loss function
def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels))


# Calculate accuracy
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# SGD Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Compute gradients
grad = tfe.implicit_gradients(loss_fn)

# Training
average_loss = 0.
average_acc = 0.
for step in range(num_steps):

    # Iterate through the dataset
    try:
        d = dataset_iter.next()
    except StopIteration:
        # Refill queue
        dataset_iter = tfe.Iterator(dataset)
        d = dataset_iter.next()

    # data (could be images, features, etc.)
    x_batch = d[0]
    # Labels
    y_batch = tf.cast(d[1], dtype=tf.int64)

    # Compute the batch loss
    batch_loss = loss_fn(neural_net, x_batch, y_batch)
    average_loss += batch_loss
    # Compute the batch accuracy
    batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
    average_acc += batch_accuracy

    if step == 0:
        # Display the initial cost, before optimizing
        print("Initial loss= {:.9f}".format(average_loss))

    # Update the variables based on SGD results
    optimizer.apply_gradients(grad(neural_net, x_batch, y_batch))

    # Log the results
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.

# Evaluate model accuracy on the test set

test_acc = accuracy_fn(neural_net, testX, testY)
print("Testset Accuracy: {:.4f}".format(test_acc))


# predict the labels
predict = tf.nn.softmax(neural_net(X_test))
y_pred_final = tf.argmax(predict, 1).numpy()
# print(np.shape(y_pred_final))
# print(type(y_pred_final))
# print (y_pred_final[100,])
# print(test_acc.numpy())

# Dump results into a *.csv file
data_pred = pd.DataFrame(y_pred_final,index = np.arange(45324, 53461, dtype=np.int), columns = ['y'])
data_pred.index.name = 'Id'
data_pred.to_csv(file_name)

print('Successfully wrote into file!')

# Dump configs (hyper-parameters,etc.) in a file
configs = np.asarray([file_name,
                        num_steps,
                        batch_size,
                        n_hidden_1,
                        n_hidden_2,
                        n_hidden_3,
                        n_hidden_4,
                        n_hidden_5,
                        n_hidden_6,
                        n_hidden_7,
                        n_hidden_8,
                        n_hidden_9,
                        test_acc.numpy()])

configs_file = pd.DataFrame(configs, index = np.arange(1, 14, dtype=np.int), columns = ['item'])
configs_file.index.name = 'Id'
configs_file.to_csv('configs.csv', mode = 'a')
