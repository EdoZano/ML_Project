import numpy as np 
import pandas as pd
import Support as sup

def check_data_types(X, y):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    return X, y

class Perceptron:
    def __init__(self, learning_rate, n):
        self.learning_rate = learning_rate
        self.n = n
        self.weights = None

    def fit(self, X, y): #POSSO UNIRLE E SEMPLIFICARE
        # X is the matrix of the features
        # y is the vector of labels
        # Inizializzazione dei pesi
        X, y = check_data_types(X, y)
        self.weights = np.zeros(X[1].shape)
        
        # Algoritmo del Perceptron
        for m in range(self.n):
            for i in range(X.shape[0]):
                linear_output = np.dot(X[i], self.weights) 
                y_predicted = np.sign(linear_output)
                
                # Aggiornamento pesi e bias se la previsione è sbagliata
                # if y[i]*linear_output <= 0
                if y_predicted != y[i]:
                    update = self.learning_rate * y[i]
                    self.weights += update * X[i]

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        y_predicted = np.sign(linear_output)
        return y_predicted

class PegasosSVM:
    def __init__(self,lambda_par, n):
        self.lambda_par = lambda_par
        self.n = n
        self.weights = None

    def fit(self, X, y):
        X, y = check_data_types(X, y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        weights_sum = np.zeros(n_features)
    
        # Algoritmo Pegasos
        for t in range(self.n ):
            # Scegli un campione casuale
            random_idx = np.random.randint(n_samples)
            x_i = X[random_idx]
            y_i = y[random_idx]

            # Calcola eta_t
            eta_t = 1 / (self.lambda_par * (t+1))

            # Verifica la condizione per l'aggiornamento
            if y_i * (np.dot(self.weights, x_i)) < 1:
                # Aggiorna w e b
                self.weights = (1 - eta_t * self.lambda_par) * self.weights + eta_t * y_i * x_i
            else:
                # Aggiorna solo i pesi
                self.weights = (1 - eta_t * self.lambda_par) * self.weights

            weights_sum += self.weights
            self.weights = weights_sum / self.n
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        return np.sign(linear_output)

class RegLogisticClass :
    def __init__(self,lambda_par,n):
        self.lambda_par=lambda_par
        self.n = n
        self.weights = None
    
    def fit(self, X, y):
        X, y = check_data_types(X, y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        t=0
        for i in range(1, self.n+1):
            # Scegli un campione casuale
            random_idx = np.random.randint(n_samples)
            X_i=X[random_idx]
            y_i=y[random_idx]

            eta_t = 1/(self.lambda_par * 1)
            # let's use z to semplify
            z = np.clip(y_i * np.dot(self.weights, X_i), -700, 700) 
            #z = y_i * np.dot(self.weights, X_i)
            #calculate the gradient of log loss
            grad = (-y_i*X_i)/(1+np.exp(z))
            #update weights
            '''Con la logistic loss, il gradiente della funzione di perdita non si "attiva" solo per i campioni mal classificati, ma fornisce un contributo anche
            per quelli classificati correttamente, anche se meno significativo. Pertanto, i due termini nell'aggiornamento (regolarizzazione e gradiente) devono 
            essere gestiti separatamente:
            Regolarizzazione: riduce l'ampiezza dei pesi a ogni iterazione.
            Gradiente della Logistic Loss: calcolato sulla base della probabilità associata al campione corrente, corregge i pesi in direzione di una 
            classificazione più accurata.'''
            self.weights = (1-eta_t*self.lambda_par)*self.weights #regolarization term
            self.weights -=  -eta_t*grad #gradient term

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        return np.sign(linear_output)

class KernelizedPerceptron:
    def __init__(self, kernel='gaussian', learning_rate=1.0, max_iter=200, sigma=1.0, c=1.0, d=2):
        self.kernel = kernel
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = None
        self.sigma = sigma
        self.c = c
        self.d = d

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                # Compute the weighted sum
                prediction = 0
                for j in range(n_samples):
                    if self.alpha[j] > 0:
                        if self.kernel == 'gaussian':
                            prediction += self.alpha[j] * self.Y[j] * sup.gaussian_kernel(X[i], X[j], self.sigma) #devi vedere come chiamare il 
                        elif self.kernel == 'polynomial':
                            prediction += self.alpha[j] * self.Y[j] * sup.polynomial_kernel(X[i], X[j], self.c, self.d)
                # Update alpha based on prediction
                if self.Y[i] * prediction <= 0:  # Misclassified
                    self.alpha[i] += self.learning_rate
    
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            prediction = 0
            for j in range(len(self.alpha)): # = n_samples
                if self.alpha[j] > 0:
                    if self.kernel == 'gaussian':
                        prediction += self.alpha[j] * self.Y[j] * sup.gaussian_kernel(X[i], self.X[j], self.sigma) #devi vedere come chiamare il kernel
                    elif self.kernel == 'polynomial':
                        prediction += self.alpha[j] * self.Y[j] * sup.polynomial_kernel(X[i], X[j], self.c, self.d)
            predictions[i] = 1 if prediction >= 0 else -1

        return predictions , self.alpha

class KernelizedPegasos:
    def __init__(self,kernel='gaussian', learning_rate=1.0, max_iter=200, sigma=1.0, c=1.0, d=2, lambda_par=0.1):
        self.kernel = kernel
        self.max_iter = max_iter
        self.learning_rate = learning_rate
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

        for _ in range(self.max_iter):
            for i in range(n_samples):
                # Calculate the prediction using the kernel
                prediction = 0
                for j in range(n_samples):
                    if self.alpha[j] > 0:  # Only consider support vectors
                        if self.kernel == 'gaussian':
                            prediction += self.alpha[j] * Y[j] * sup.gaussian_kernel(X[i], X[j], self.sigma)
                        elif self.kernel == 'polynomial':
                            prediction += self.alpha[j] * Y[j] * sup.polynomial_kernel(X[i], X[j], self.c, self.d)

                # Update alpha based on prediction
                if Y[i] * prediction <= 1:  # Misclassified or on the margin
                    self.alpha[i] += self.learning_rate * (1 - Y[i] * prediction)
                # Regularization step
                self.alpha[i] = max(0, self.alpha[i] - self.learning_rate * self.lambda_param * self.alpha[i]) # prevent overfitting and to ensure the weights don't grow too large. 
        # Identify support vectors
        self.support_vector_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = X[self.support_vector_indices]


    def predict(self, X) :
        predictions = []
        for x in X:
            prediction = 0
            for j in self.support_vector_indices:
                if self.kernel == 'gaussian':
                    prediction += self.alpha[j] * sup.gaussian_kernel(x, self.support_vectors[j], self.sigma)  # Alpha for support vector
                elif self.kernel == 'polynomial':
                    prediction += self.alpha[j] * sup.polynomial_kernel(x, self.support_vectors[j], self.c, self.d)
            predictions.append(np.sign(prediction))  # Sign of prediction
        return np.array(predictions)