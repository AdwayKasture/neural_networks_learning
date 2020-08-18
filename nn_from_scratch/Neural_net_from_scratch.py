import numpy as np

#back prpogation to be implemented
'''
data generator to test basic model on
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

print(X,y)

Architecture of nn
1 2 
1 2 3 
1 2 3 4 5
1 2 3



X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
'''


class Neural_net :
    def __init__ (self):
        self.inp = Layer_Dense(2,3)
        self.l1 = Layer_Dense(3,5)
        self.l2 = Layer_Dense(5,3)
        self.out = Output_layer(3)
    def test(self,inpp):
        self.inp.forward_begin(inpp)
        self.l1.forward(relu(self.inp.output,3))
        #print(self.l1.output)
        self.l2.forward(self.l1.output)
        #print(self.l2.output)
        self.out.forward(self.l2.output)
        print(self.out.answer)

    def display(self):
        print(self.inp.weights)
        print(self.l1.weights)
        print(self.l2.weights)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def forward_begin(self,inputs):
        weights = [0.1,0.1,0.1]
        self.output = np.dot(inputs,self.weights)
            
        
#to manually set weights instead of random t oobserrve        
class Layer_Dense2:
    def __init__(self,weightlist,n_inp,n_neu):
        self.weights = np.asarray(weightlist)
        self.biases = np.zeros((1, n_neu))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def forward_begin(self,inputs):
        weights = [0.1,0.1,0.1]
        self.output = np.dot(inputs,self.weights)
    def show(self):
        print(self.weights,self.biases)
    def change_weights(self,x0,y0,val):
        self.weights[x0 , y0] = val

#activation function
def relu(inpp,i_max):
    print(inpp)
    for i in range(i_max):
            if inpp[i] < 0 :
                inpp[i] = 0.0
    print(inpp)
    return inpp


#output layer with inherent soft max
class Output_layer():
    def __init__(self,numb):
        self.array = np.ones((numb))
    def forward(self,inpp):
        for i in range(len(inpp[0])):
            if inpp[0][i] == max(inpp[0]):
                k = i
                break 
        self.answer =  k


#simple loss for gradient descent/back prop
def loss(out, Y): 
    s =(np.square(out-Y)) 
    s = np.sum(s)/len(y) 
    return(s) 


#testing / output
n1 = Neural_net()
n1.test([1,1])
#n1.test([1,0])
#n1.test([0,1])
#n1.test([0,0])
