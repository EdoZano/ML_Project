import numpy as np  # type: ignore
import pandas as pd # type: ignore
import Support as sup

def check_data_types(X, y):
    ''' 
    Ensures that X and y are NumPy arrays. If X is a DataFrame or y is a Series, they are converted.
    This helps maintain consistency in data formats during training.'''
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    return X, y

class Perceptron:
    ''' 
    Standard Perceptron algorithm for binary classification.
    Parameters:
    - learning_rate (float): Step size for weight updates.
    - n (int): Number of iterations (epochs) during training.'''

    def __init__(self, learning_rate, n):
        self.learning_rate = learning_rate
        self.n = n
        self.weights = None



    def fit(self, X, y): 
        ''' 
        Trains the Perceptron model by iterating over the dataset and updating weights
        based on misclassifications.
        X is the matrix of the features
        y is the vector of labels'''
        X, y = check_data_types(X, y)
        self.weights = np.zeros(X.shape[1])
        
        # Perceptron algorithm
        for m in range(self.n):
            for i in range(X.shape[0]):
                linear_output = np.dot(X[i], self.weights) 
                y_predicted = np.sign(linear_output)
                # weight update if the prediction is incorrect
                if y_predicted != y[i]:
                    update = self.learning_rate * y[i]
                    self.weights += update * X[i]

    def predict(self, X):
        ''' Predicts class labels for input data.'''
        linear_output = np.dot(X, self.weights)
        y_predicted = np.sign(linear_output)
        return y_predicted #output +1 or -1

class PegasosSVM:
    '''
    Pegasos algorithm for Support Vector Machines (SVM).

    Parameters:
    - lambda_par (float): Regularization parameter controlling weight decay.
    - n (int): Number of training iterations.'''
    def __init__(self,lambda_par, n):
        self.lambda_par = lambda_par
        self.n = n
        self.weights = None

    def fit(self, X, y):
        '''
        Trains Pegasos SVM using stochastic updates on randomly selected samples.
        X is the matrix of the features
        y is the vector of labels'''
        X, y = check_data_types(X, y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        weights_sum = np.zeros(n_features)
    
        # Pegasos Algorithm
        for t in range(self.n ):
            # Shuffle data for each epoch
            random_idx = np.random.randint(n_samples)
            x_i = X[random_idx]
            y_i = y[random_idx]

            # eta_t
            eta_t = 1 / (self.lambda_par * (t+1))

            # Check margin condition
            if y_i * (np.dot(self.weights, x_i)) < 1:
                self.weights = (1 - eta_t * self.lambda_par) * self.weights + eta_t * y_i * x_i
            else:
                self.weights = (1 - eta_t * self.lambda_par) * self.weights

            weights_sum += self.weights # Accumulate weights for averaging
        self.weights = weights_sum / self.n  # Compute the average of the weights
    
    def predict(self, X):
        '''Predicts class labels for input data.'''
        linear_output = np.dot(X, self.weights)
        return np.sign(linear_output)

class RegLogisticClass :
    '''
    Logistic Regression with L2 regularization.
    Parameters:
    - lambda_par (float): Regularization strength to prevent overfitting.
    - n (int): Number of training iterations.'''
    def __init__(self,lambda_par,n):
        self.lambda_par=lambda_par
        self.n = n
        self.weights = None
    
    def fit(self, X, y):
        '''
        Trains the model using stochastic gradient descent (SGD)
        X is the matrix of the features
        y is the vector of labels'''
        X, y = check_data_types(X, y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for i in range(1, self.n+1):
            # Shuffle data for each epoch
            random_idx = np.random.randint(n_samples)
            X_i=X[random_idx]
            y_i=y[random_idx]

            eta_t = 1/(self.lambda_par * i)
            # let's use z to semplify
            z = np.clip(y_i * np.dot(self.weights, X_i),-700,700) 
            #calculate the gradient of log loss
            grad = (-y_i*X_i)/(1+np.exp(z)) +self.lambda_par*self.weights
            #update weights
            '''Con la logistic loss, il gradiente della funzione di perdita non si "attiva" solo per i campioni mal classificati, ma fornisce un contributo anche
            per quelli classificati correttamente, anche se meno significativo. Pertanto, i due termini nell'aggiornamento (regolarizzazione e gradiente) devono 
            essere gestiti separatamente:
            Regolarizzazione: riduce l'ampiezza dei pesi a ogni iterazione.
            Gradiente della Logistic Loss: calcolato sulla base della probabilità associata al campione corrente, corregge i pesi in direzione di una 
            classificazione più accurata.'''
            self.weights -=  -eta_t*grad # weights update

    def predict(self, X):
        ''' Predicts class labels for input data.'''
        linear_output = np.dot(X, self.weights)
        return np.sign(linear_output)

class KernelizedPerceptron:
    '''
    Perceptron algorithm extended with kernel functions.

    Parameters:
    - kernel (str): Type of kernel ('gaussian' or 'polynomial').
    - learning_rate (float): Step size for updates.
    - max_iter (int): Number of training iterations.
    - sigma (float): Bandwidth for Gaussian kernel.
    - c (float): Coefficient for polynomial kernel.
    - d (int): Degree of polynomial kernel.'''
    def __init__(self, kernel='gaussian', learning_rate=1.0, max_iter=200, sigma=1.0, c=1.0, d=2):
        self.kernel = kernel
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = None
        self.sigma = sigma
        self.c = c
        self.d = d
        self.X = None
        self.Y = None

    def fit(self, X, Y):
        '''Trains the Kernelized Perceptron using the selected kernel.'''
        self.X = X
        self.Y = Y
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)

        if self.kernel == 'gaussian':
            kernel_matrix = sup.gaussian_kernel_vectorized(X, X, self.sigma)
        elif self.kernel == 'polynomial':
            kernel_matrix = sup.polynomial_kernel_vectorized(X, X, self.c, self.d)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                prediction = np.sum(self.alpha * self.Y * kernel_matrix[i, :])
                if self.Y[i] * prediction <= 0:  # Misclassified
                    self.alpha[i] += self.learning_rate
    
    def predict(self, X):
        '''Predicts class labels using support vectors and kernel similarity.'''
        # Compute the kernel matrix between test points and training points
        if self.kernel == 'gaussian':
            kernel_matrix = sup.gaussian_kernel_vectorized(X, self.X, self.sigma)
        elif self.kernel == 'polynomial':
            kernel_matrix = sup.polynomial_kernel_vectorized(X, self.X, self.c, self.d)
        
        predictions = np.sum(self.alpha * self.Y * kernel_matrix, axis=1)
        return np.where(predictions >= 0, 1, -1), self.alpha

class KernelizedPegasos:
    '''
    Kernelized Pegasos algorithm for Support Vector Machines (SVM).
    Parameters:
    - kernel (str): Type of kernel to use ('gaussian' or 'polynomial').
    - max_iter (int): Number of training iterations.
    - sigma (float): Bandwidth parameter for the Gaussian kernel.
    - c (float): Coefficient term for the polynomial kernel.
    - d (int): Degree of the polynomial kernel.
    - lambda_par (float): Regularization parameter that controls weight decay.'''
    def __init__(self,kernel='gaussian', max_iter=200, sigma=1.0, c=1.0, d=2, lambda_par=0.1):
        self.kernel = kernel
        self.max_iter = max_iter
        self.alpha = None
        self.sigma = sigma
        self.c = c
        self.d = d
        self.lambda_par = lambda_par
        '''Purpose: Stores the data points that are the support vectors.
        Support Vectors: These are the data points that directly influence the decision boundary of the SVM. In a kernel SVM, support vectors are critical because
                 the decision boundary is defined by the points for which the Lagrange multipliers 𝛼  are non-zero.
        Usage: During prediction, the kernel function is computed only between new data points and the support vectors. This helps make predictions more efficient 
                because you don't need to compute the kernel with all training data points, just the support vectors.''' 
        self.support_vectors = None
        self.support_vector_indices = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.Y = Y
        if self.kernel == 'gaussian':
            kernel_matrix = sup.gaussian_kernel_vectorized(X, X, self.sigma)
        elif self.kernel == 'polynomial':
            kernel_matrix = sup.polynomial_kernel_vectorized(X, X, self.c, self.d)
        for t in range(1, self.max_iter+1):
            #randomly select an index i_t
            i_t = np.random.randint(0, n_samples)
            #compute margin condition
            margin = (self.Y[i_t]/(self.lambda_par * t)) * np.sum(self.alpha * self.Y * kernel_matrix[i_t, :])
            if  margin < 1:
                self.alpha[i_t] += 1 
        # Store support vectors 
        self.support_vector_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = X[self.support_vector_indices]
        


    def predict(self, X) :
        '''Predicts class labels using support vectors and kernel similarity'''
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        # Compute the kernel matrix between test points and support vectors
        if self.kernel == 'gaussian':
            kernel_matrix = sup.gaussian_kernel_vectorized(X, self.support_vectors, self.sigma)
        elif self.kernel == 'polynomial':
            kernel_matrix = sup.polynomial_kernel_vectorized(X, self.support_vectors, self.c, self.d)
        #calculate predictions using support vectors
        for i in range(n_samples):
            prediction = np.sum(self.alpha[self.support_vector_indices] * self.Y[self.support_vector_indices] * kernel_matrix[i, :])
            predictions[i]=np.sign(prediction)
        return np.array(predictions), self.alpha