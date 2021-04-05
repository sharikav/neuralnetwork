'''
Design of a Neural Network from scratch
4 layers
input layer->9 neurons
hidden layer1 -> 11 neurons
hidden layer2 -> 5 neurons
output layer -> 1 neuron
weights and bias are a list of 3 lists having 11,5 and 1 values respectively
sigmoid activation function
MSE loss function
batch_size = 16 #number of data points to process in each batch
epochs = 100 #number of epochs for the training
learning_rate=0.4 #value of the learning rate
validation_split=0.2#train_test_split as the usual 0.3 makes the training data too less for a good accuracy

additional components-
I plotted the Validation Loss Graph
I used shuffling from sklearn to get better results, not a necessary component 
'''
import numpy as np
import pandas as pd
from sklearn.utils import shuffle#not mandatory accessory
from sklearn.model_selection import train_test_split# train_test_split
from matplotlib import pyplot as plt#plotting the graph
batch_size = 16 #number of data points to process in each batch
epochs = 100 #number of epochs for the training
learning_rate=0.4 #value of the learning rate
validation_split=0.2#train_test_split

def plot_history(history):
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    n = 100
    plt.plot(range(100)[:n], history[2][:n], label='train_loss')
    plt.plot(range(100)[:n], history[3][:n], label='test_loss')
    plt.title('train & test loss')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()



def preprocess(file):
    """
    We observe that community column has no null values 
    Age column has 7 null values that we fill using the mean value
    Weight column has 11 null values that we fill using the mean value
    Delivery phase has 4 null values that we fill with its respective previous value
    HB column has 19 null values that is filled using the mean value
    IFA columnn has no null values 
    BP column has 15 null values which we fill with the mean value
    Education has 3 null values, however since every other column has the value 5, we fill these with the same value
    Residence column has 2 null values which we fill with the previous values
    We also observe that 72 rows have result 1 and 24 rows have result 0, i.e. it is an unbalanced dataset
    """
    df = pd.read_csv(file)
    df['Age']=df['Age'].fillna(df['Age'].mean())
    df['Weight']=df['Weight'].fillna(df['Weight'].mean())
    df['Delivery phase']=df['Delivery phase'].fillna(method='ffill')
    df['HB']=df['HB'].fillna(df['HB'].mean())
    df['BP']=df['BP'].fillna(df['BP'].mean())
    df['Education'].fillna(5.0, inplace=True)
    df['Residence']=df['Residence'].fillna(method='ffill')
      
    return df

def activation(z, derivative=False):
    """
    I used a Sigmoid activation function:
    Unlike the normal neural networks where the dafault is ReLU.
    It handles two modes: normal and derivative mode.
    Applies a pointwize operation on vectors
    
    Parameters:
    ---
    z: pre-activation vector at layer l
        shape (n[l], batch_size)

    Returns: 
    pointwize activation on each element of the input z
    """
    if derivative:        
        return activation(z)*(1-activation(z))
    else:
        return 1 / (1 + np.exp(-z))



def cost_function(y_true, y_pred):
    """
    Computes the Mean Square Error between truth values and a prediction values
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost: a scalar value representing the loss
    
    """
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost

def cost_function_der(y_true, y_pred):
    """
    Computes the derivative of the loss function w.r.t the activation of the output layer
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost_prime: derivative of the loss w.r.t. the activation of the output
    shape: (n[L], batch_size)    
    """
    cost_prime = y_pred - y_true
    return cost_prime
    

class NN:
    '''X and Y are 2D arrays'''
    def __init__(self, size, seed=42):
        """
        Instantiate the weights and biases of the network
        weights and biases are attributes of the NeuralNetwork class
        They are updated during the training
        Since we have a total of 4 layers, having 9,11,5,1 neurons respectively, bias and weights is a list of 3 lists of lengths 11, 5 and 1.
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]
    
    def forward(self, input):
        '''
        Perform a feed forward computation 

        Parameters
        ---
        input: data to be fed to the network with
        shape: (input_shape, batch_size)

        Returns
        ---
        a: ouptut activation (output_shape, batch_size)
        pre_activations: list of pre-activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l
        activations: list of activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l

        '''
        a = input
        pre_activations = []
        activations = [a]
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a  = activation(z)
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations
    
    def compute_deltas(self, pre_activations, y_true, y_pred):
        """
        Computes a list containing the values of delta for each layer using 
        a recursion
        Parameters:
        ---
        pre_activations: list of of pre-activations. each corresponding to a layer
        y_true: ground truth values of the labels
        y_pred: prediction values of the labels
        Returns:
        ---
        deltas: a list of deltas per layer
        
        """
        d = cost_function_der(y_true, y_pred) * activation(pre_activations[-1], derivative=True)
        deltas = [0] * (len(self.size) - 1)
        deltas[-1] = d
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * activation(pre_activations[l], derivative=True) 
            deltas[l] = delta
        return deltas
    
    def backpropagate(self, deltas, pre_activations, activations):
        """
        Applies back-propagation and computes the gradient of the loss
        w.r.t the weights and biases of the network

        Parameters:
        ---
        deltas: list of deltas computed by compute_deltas
        pre_activations: a list of pre-activations per layer
        activations: a list of activations per layer
        Returns:
        ---
        dW: list of gradients w.r.t. the weight matrices of the network
        db: list of gradients w.r.t. the biases (vectors) of the network
    
        """
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db
    def fit(self,X,y):
        '''
		Function that trains the neural network by taking x_train and y_train samples as input
		It trains the network using the gradients computed by back-propagation
        Splits the data in train and validation splits
        Processes the training data by batches and trains the network using batch gradient descent

        Parameters:
        ---
        X: input data
        y: input labels
        
        Returns:
        x_test and y_test    
	    '''
        
        history_train_losses = []
        
        history_test_losses = []
        
        #train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X.T, y.T, test_size=validation_split, )
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

        epoch_iterator = range(epochs)
        #iteration 
        for e in epoch_iterator:
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size ) - 1

            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T
            #creating batches
            batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            
            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]#delta weights
            db_per_epoch = [np.zeros(b.shape) for b in self.biases]#delta biases 
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                #forwarding
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                #computing delta
                deltas = self.compute_deltas(pre_activations, batch_y, batch_y_pred)
                #back propogation
                dW, db = self.backpropagate(deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)
                #for plotting and calculating loss 
                train_losses=[]
                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)
                batch_y_test_pred = self.predict(x_test)
                test_losses=[]
                test_loss =cost_function(y_test, batch_y_test_pred)
                test_losses.append(test_loss)
                


            # weight update
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_test_losses.append(np.mean(test_losses))
            
        history = {'epochs': epochs,
                   'train_loss': np.mean(history_train_losses), 
                   'test_loss': np.mean(history_test_losses),
                  }
        print(history)
        return x_test,y_test,history_train_losses,history_test_losses
    
    def predict(self,a):
        """
		The predict function performs a simple feed forward of weights
		and outputs yhat values 
		
        Use the current state of the network to make predictions

        Parameters:
        ---
        a: input data, shape: (input_shape, batch_size)

        Returns:
        ---
        yhat: vector of output predictions
        """
        for w, b in zip(self.weights, self.biases):

            z = np.dot(w, a) + b
            a = activation(z)
        yhat = (abs(a) > 0.735).astype(int)
        return yhat
    
    
    def CM(self,y_test,y_test_obs):
        '''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.735):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
		
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
		
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        a = (tn+tp)/(tn+fp+fn+tp)
		
        print("\nConfusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {round(p*100)}%")
        print(f"Accuracy : {round(a*100)}%")
        print(f"Recall : {round(r*100)}%")
        print(f"F1 SCORE : {round(f1*100)}%")
        print(f"True Values : {y_test}")
        print(f" Predicted Values : {y_test_obs}")
			
if __name__ == "__main__":
    df = preprocess('LBW_Dataset.csv')
      
    X=df.iloc[:,:-1]
    X=X.to_numpy()# Dimensions->(96x9) 
    
    y=df.iloc[:,-1]
    y=y.to_numpy()
    y=np.expand_dims(y, 1)#Dimensions->(96x1)
    
    
    model = NN([9,11,5, 1],seed=0)
    """
    Creates a neural network of 4 layers:
        Input layer-> 9 neurons
        Hidden layer 1-> 11 neurons
        Hidden layer 2-> 5 neurons
        Output layer-> 1 neuron
    """
    
    history = model.fit(X=X.T,y=y.T)
    '''
    History returns the following lists :     
    history[0] : returns the test samples (X_test)
    history[1] : return the test output labels (Y_test) 
    '''
    y_test= history[1][0]
    y_pred= model.predict(history[0])
    model.CM(y_test.flatten(),y_pred[0].flatten())
    plot_history(history)



