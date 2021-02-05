import numpy as np


'''
A simple neural net using numpy
made after refering sentex video and https://victorzhou.com/blog/intro-to-neural-networks/
the problem is taken from there
activation function sigmoid :
input layer (2)
hidden layer(2)
output layer(1)
'''

def relu(inpp):
    inpp =np.maximum(inpp,0)
    return inpp

def deriv_relu(inpp):
    inpp[inpp > 0 ] = 1
    inpp[inpp <=0 ] = 0
    return inpp

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))


class layer :
    def __init__(self,inp,out):
        self.weights = 0.1 * np.random.randn(inp, out)
        self.biases = np.zeros((1, out))
        self.inp = inp
    def forward(self, inputs):
        #store out_put in .output feed to next layer during feed forward
        self.output = sigmoid(np.dot(inputs, self.weights) + self.biases)
        return self.output



class model :
    def __init__(self):
        #input -> hidden layer l1 
        self.l1 = layer(2,2)
        # hidden layer -> output layer 
        self.l2 = layer(2,1)
    def predict(self,x):
        #input -> hidden layer and store in output_1
        self.output_1 = self.l1.forward(x)
        #hidden layer -> output final output in output_2
        self.output_2 = self.l2.forward(self.output_1)
        #return the output
        return self.output_2


    def back_prop(self,x,y,learn_r):
        self.predict(x)
        #last layer ,partial deriv wrp all params
        d_L_d_ypred = -2 *(y - self.output_2)
        d_ypred_d_w = self.output_1 *deriv_sigmoid(self.output_2)
        d_ypred_d_b = deriv_sigmoid(self.output_2)
        d_ypred_d_ll1 = self.l2.weights * deriv_sigmoid(self.output_2)
        #second last layer
        d_output1_d_w = x * deriv_sigmoid(l1.output)
        d_output1_d_b = deriv_sigmoid(l1.output)
        #update_weights l1
        self.l1.weights = self.l1.weights - learn_r * d_L_d_ypred * np.transpose(d_ypred_d_ll1) * d_output1_d_w
        self.l1.biases = self.l1.biases - learn_r * d_L_d_ypred * np.transpose(d_ypred_d_ll1) * d_output1_d_b
        #update weights l2
        self.l2.weights = self.l2.weights - learn_r * d_L_d_ypred * np.transpose(d_ypred_d_w)
        self.l2.biases = self.l2.biases - learn_r * d_L_d_ypred * np.transpose(d_ypred_d_b)

        
    def mass_predict(self,x):
        #predict for the batch 
        return(np.apply_along_axis(self.predict, 1, x))


#initialise model
m1 = model()

#input data x
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

#input data Y
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])


#function to train the model  on data
def train_model(modell,data_x,data_y):
    learn_rate = 0.1
    epochs = 1000
    #number of iterations 
    for epoch in range(epochs):
        for x,Y in zip(data_x,data_y):
            #back propogation
            modell.back_prop(x,Y,learn_rate)
            #check loss for batch after every 10 epoch should decrease in each
        if (epoch % 10) == 0:
            #predict for batch
            y_pred = modell.mass_predict(data_x)
            #reshape to compare (4,1,1) -> (4,)
            y_pred = y_pred.reshape(-1)
            #MSE 
            loss = np.mean(np.square((y_pred - data_y)))
            print("Epoch %d loss: %.3f" % (epoch, loss))


#testing inital for (-12,-14) should return 1 after training , before most likely will be near 0.5
m1.predict([-12,-14])#value stored in output_2

print(m1.output_2)

#training model
train_model(m1,data,all_y_trues)

#to check if model trained should almost be  1
m1.predict([-12,-14])
print(m1.output_2)
