import tensorflow as tf
import pandas as pd
import numpy as np
tf.enable_eager_execution()



# dataset generator simple linear function
def generate_dataset():
    x_batch = np.linspace(0, 2, 100)
    y_batch = 1.5 * x_batch + np.random.randn(*x_batch.shape) * 0.2 + 0.5
    return x_batch, y_batch

x,y = generate_dataset()



#model for linear regression
class MyModel(tf.Module):
    def __init__(self):
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(0.0)
        self.b = tf.Variable(0.0)

    # feed forward for the model to execute for given x
    def __call__(self, x):
        return self.w * x + self.b


#calculate average for the batch 
def loss(Y,y_pred):
    return tf.reduce_mean(tf.square(Y - y_pred))

#for each epoch we will perform the following
def train_step(model, x, y, learning_rate):

    #bread and butter for gradient descent TF implementation for managing parameters
    with tf.GradientTape() as t:
    # Trainable variables are automatically tracked by GradientTape
        # loss in current epoch calls model to predict output 
        current_loss = loss(y, model(x))

  # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

  # Subtract the gradient scaled by the learning rate w = w-w*dw
  #update weights 
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)

#invoke model , set epochs
model = MyModel()
epochs = range(10000)

# Define a training loop
def training_loop(model, x, y):

    for epoch in epochs:
        # Update the model with the single giant batch
        train_step(model, x, y, learning_rate=0.001)
        #current loss in epoch
        current_loss = loss(y, model(x))
        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
            (epoch, model.w.numpy(),model.b.numpy(), current_loss))


# Do the training
training_loop(model, x, y)


# test 
k1 = model(1)
#k1 should be ~2  as 1.5*1 + 0.5 == 2
print(k1)
