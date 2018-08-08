import numpy as np
import matplotlib.pyplot as plt

class feed_forward:
    
    def __init__(self,input_dim=10,task='regression',non_linearity='tanh',optimizer='sgd',num_classes=None):
        
        self.input_dim=input_dim
        self.non_linearity=non_linearity
        self.task=task
        self.optimizer=optimizer
        self.num_classes=num_classes
        
        if self.task=='classification':      
            self.loss='cross_entropy'
            
        elif self.task=='regression':
            self.loss='mean_squared_error' 
            
        else:
            raise Exception('Only classification and regression supported!')
            
        self.weights=[]
        self.biases=[]
        
    def initializer(self,shape):
        #feel free to have more complicated initializers, here we just use a normal dist initializer. 
        return np.random.normal(0,0.1,shape)
    
    def add(self,num_neurons): #method to add a new layer
        
        if self.weights==[]: #this is the first layer connected to input
            self.weights.append(self.initializer([self.input_dim,num_neurons]))
            self.biases.append(self.initializer([num_neurons,1]))
        else:
            self.weights.append(self.initializer([self.weights[-1].shape[1],num_neurons])) #previous weight second dim x num_neurons
            self.biases.append(self.initializer([num_neurons,1]))
            
            
    def predict(self,input_data,train='0'): #if train is 0, only returns final output, else all the activations of the layers.
        
        num_examples=input_data.shape[0]
        
        output=[]                 #will hold all the neuron outputs after non-linearity is applied.
        output.append(input_data) #first appending the input data, useful for backprop
        a1=np.dot(input_data,self.weights[0])+ np.ones([num_examples,1])*self.biases[0].T #also written as X*w + np.tile(bias,[samples,1])
        z1=self._non_linear(a1)
        output.append(z1) 
        
        for idx in range(1,len(self.weights)): #one weight has already been used, let us use the other ones
            
            out=np.dot(output[-1],self.weights[idx]) + np.ones([num_examples,1])*self.biases[idx].T 
            
            if idx!=len(self.weights)-1: #Only in the last layer, don't apply the traditional non linearity. We may apply softmax instead.
                output.append(self._non_linear(out))    
            else:
                output.append(out)
               
        if self.task=='classification':      
            
            o=output[-1]
            vals=np.exp(o-np.tile(np.max(o,axis=1),(o.shape[1],1)).T) #max value subtracted for numerical stability.
            sums=np.sum(vals,axis=1)
            prob=vals/(np.tile(sums,(o.shape[1],1)).T)
            output[-1]=prob #last output is the transformed softmax
            
            if train=='0':
                return output[-1] #these contain the probabilties
            else:
                return output
        
        elif self.task=='regression':
            if train=='0':
                return output[-1]
            else:
                return output
    
    def _non_linear(self,x,deriv='false'): # _ before indicates private method
        
        if deriv=='false':
            
            if self.non_linearity=='relu':
                return np.maximum(x,0)
            
            elif self.non_linearity=='tanh':
                return np.tanh(x)
            
        else: #we want the derivative, not the activation
            
            if self.non_linearity=='relu':
                
                return 1*np.array(x>0)
            
            elif self.non_linearity=='tanh':
                
                return 1-np.square(x) #note that x inputted is already the tanh of the activation, so we do only square, and not square(tanh(x))
            
    def convert_to_one_hot(self,t):
        #t is [0,2,1] etc, purely class indices.
        labels=t.reshape(-1,1) #incase it is of shape (x,)
        t = np.eye(self.num_classes)[labels.reshape(-1)]
        return t
    
    def compute_loss(self,x,t):
        
        if self.loss=='mean_squared_error':
            
            output=self.predict(x)
            mean_squared_error_value=np.mean(np.square(output-t))
            return mean_squared_error_value
            
        elif self.loss=='cross_entropy':
            
            labels=self.convert_to_one_hot(t)
            probs=self.predict(x) #the probabilities
            cross_entropy_loss_value=np.mean(np.sum(-np.log(probs)*labels,axis=1))
            return cross_entropy_loss_value
        
    def get_gradients(self,x,t):
        grad_weights=[]
        grad_biases=[]
        
        activations=self.predict(x,train='1') #calling it with train='1', so function returns all the layer activations
        
        if self.loss=='mean_squared_error':
            
            delta=(activations[-1]-t) #y-t 
            
        elif self.loss=='cross_entropy':
            
            delta=activations[-1]- self.convert_to_one_hot(t) #y_i - t_i cross_entr gradient, computed by converting  t to one hot labels
            
        #first, the gradient of the weights connecting to the output
        
        grad_weights.append(np.matmul(activations[-2].T ,delta)/x.shape[0]) #we divide by number of samples in batch for av. loss!
        grad_biases.append(np.dot(delta.T,np.ones([x.shape[0],1]))/x.shape[0])

        for idx in range(1,len(self.weights)):
            
            delta_prev= np.dot(delta,self.weights[len(self.weights)-idx].T)*self._non_linear(activations[-1-idx],deriv='true')                             #dl/dlayer_prev. Refer CM Bishop.
            
            grad_weights.insert(0,np.matmul(activations[-2-idx].T ,delta_prev)/x.shape[0]) #adding to the first, since we are going backwards in backprop
            grad_biases.insert(0,np.dot(delta_prev.T,np.ones([x.shape[0],1])/x.shape[0]))
        
            delta=delta_prev  #now set delta to delta_prev, as now this layer delta is used for previous layer's gradients.
            
        return [grad_weights,grad_biases]
    
    def train_network(self,x,t,num_iterations=100,learning_rate=0.1):
        
        t=t.reshape(-1,1) #Incase it is of shape (x,), we add an extra dimension for compute.
        
        for idx in range(num_iterations):
            [g_weights,g_biases]=self.get_gradients(x,t)
            
                
            if self.optimizer=='sgd':
                for cnt,k in enumerate(zip(g_weights,g_biases)):
                    
                    self.weights[cnt]-=k[0]*learning_rate #gradient descent for weight
                    self.biases[cnt]-=k[1]*learning_rate  #GD for bias
            
            if idx%100==0:
                loss_val=self.compute_loss(x,t)
                print('The training {} loss is now {}'.format(self.loss,loss_val))
                learning_rate=learning_rate*0.99 #learning rate decay
                
#Let's generate a dataset with circular decision boundary, visualize it and test the network!
                
thresh=0.8 #this will be the square of the radius of the circle

train_input=1-2*np.random.rand(1000,2) #uniformly distributed over a box
y=[]
for k in list(train_input):
    if np.linalg.norm(k,2)>thresh: #if outside a circle, set to class label 1
        y.append(1)
    else:
        y.append(0)

y=np.array(y)

#Let us build the feed forward network. 

b=feed_forward(input_dim=2,task='classification',num_classes=2,non_linearity='relu')                  
b.add(10)  #10 neuron layer
b.add(5)
b.add(2) #2 output neurons, for 2 classes. If regression, you can use 1. 

b.train_network(train_input,y,num_iterations=5000,learning_rate=0.1)  #we do not use minibatches or a validation set, feel free to modify!

#Let us now make a new test set!

test_input=1-2*np.random.rand(1000,2) 

predict_probs=b.predict(test_input) #predict the probabilties for the input data
class_outputs=np.argmax(predict_probs,axis=1) #taking maximum probability

plt.scatter(test_input[:,0],test_input[:,1],c=class_outputs)          #visualize the output of the network on test set.
circle1=plt.Circle((0,0),np.sqrt(thresh),fill=False,color='r')        #plot the circular decision boundary
plt.gcf().gca().add_artist(circle1)



#
