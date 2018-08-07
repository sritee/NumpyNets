# NumpyNets

A modular, keras like class based implementation for building neural networks using numpy. 


### Regression: 

regression_net=feed_forward(input_dim=5, task='regression', non_linearity='tanh')                  
regression_net.add(10)  #add a layer with 10 neurons, plus a bias neuron  
regression_net.add(8)  
regression_net.add(1)  #output  
regression_net.predict(np.random.rand(1,5))   #make a prediction  
regression_net.train_network(x,t,iterations=100)   #train  

### Classification:

class_net=feed_forward(input_dim=10, task='classification', num_classes=3, non_linearity='relu')        
class_net.add(10)   
class_net.add(3) #number of classes  
class_net.predict(np.random.rand(1,10)) #predict the probabilities.  


