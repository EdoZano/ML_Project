import numpy as np 
import pandas as pd
import itertools as it

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


def Zero_One_Loss(Y_true, Y_hat):
    numbermiss = np.mean(Y_hat != Y_true)
    return numbermiss


def Polynomial_exp(X_train,X_test, column_names): 
    X_train_df=pd.DataFrame(X_train, columns=column_names)
    X_test_df=pd.DataFrame(X_test, columns=column_names)
  
    for i in column_names:
        X_train_df[f'{i}^2']=X_train_df[i]**2
        X_test_df[f'{i}^2']=X_test_df[i]**2
   
    for i in range(len(column_names)):
        for j in range(i+1,len(column_names)):
            col1 = column_names[i]
            col2 = column_names[j]
            X_train_df[f'{col1}*{col2}'] = X_train_df[col1] * X_train_df[col2]
            X_test_df[f'{col1}*{col2}'] = X_test_df[col1] * X_test_df[col2]
    X_poly_train = X_train_df.values
    X_poly_test = X_test_df.values
    return X_train_df , X_test_df

def k_Cross_Validation(model_class, X, y, params, k):
    """    
    model_class: la classe del modello (Perceptron, Pegasos, etc.)
    X: DataFrame delle feature
    y: Array degli outcome
    params: parametri da passare al modello
    k: numero di fold    
    Restituisce la lista delle accuratezze per ciascun fold.
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Shuffle per mescolare i dati

    scores= []

    for i in range(k):
        start = i*fold_size
        if i < k-1 :
            end = start + fold_size
        else :
            end = n_samples
        
        #calcoliamo indici di validation e trainng set
        validation_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, validation_idx)
        #troviamo validaation e training set
        X_train, X_val = X.iloc[train_idx], X.iloc[validation_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[validation_idx]
        #chiamiamo e fittiamo nostro modello
        model = model_class(**params)
        model.fit(X_train, y_train)
        #cacoliamo ciò di cui abbiamo bisogno
        y_hat = model.predict(X_val)
        loss =  Zero_One_Loss (y_val, y_hat, y_val.shape)

        scores.append(loss)
    
    return np.mean(scores)


def tuning_par(model, X, y, params, k): #richiede X dataframe, no array
    '''
    model_class: la classe del modello (Perceptron, Pegasos, etc.)
    X: DataFrame delle feature
    y: Array dei target
    param_grid: dizionario con i parametri da esplorare (es. {'eta': [0.01, 0.1], 'n_iter': [100, 200]})
    k: numero di fold per la cross-validation (es. 5)
    
    Restituisce il miglior set di parametri e la corrispondente accuratezza media.
    '''
    
    best_param = None
    best_loss=np.inf
    #creiamo tuple per ogni possibile combinazione
    keys, values = zip(*params.items())
    param_combinations = [dict(zip(keys, v)) for v in it.product(*values)]

    for i in param_combinations:
        loss = k_Cross_Validation(model, X, y, i, k=k)

        if(loss<best_loss):
            best_loss=loss
            best_param=i
    
    return best_loss, best_param


def gaussian_kernel(x1, x2, sigma=1.0):
    """
    Parameters:
    x1: First input vector.
    x2: Second input vector.
    sigma: Standard deviation of the Gaussian distribution.
    kernel: The Gaussian kernel value between x1 and x2.
    """
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

def polynomial_kernel(x1, x2, c=1.0, d=3):
    """
    x1: First input vector.
    x2: Second input vector.
    c: Constant term in the polynomial kernel.
    d: Degree of the polynomial.
    """
    return (np.dot(x1, x2) + c) ** d    

class KernelizedPerceptron:
    def __init__(self, kernel='gaussian', learning_rate, max_iter=1000, sigma, c, d):
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
                            prediction += self.alpha[j] * self.y[j] * gaussian_kernel(X[i], X[j], self.sigma) #devi vedere come chiamare il 
                        elif self.kernel == 'polynomial':
                            prediction += self.alpha[j] * self.y[j] * polynomial_kernel(X[i], X[j], self.c, self.d)
                # Update alpha based on prediction
                if self.y[i] * prediction <= 0:  # Misclassified
                    self.alpha[i] += self.learning_rate
    
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            prediction = 0
            for j in range(len(self.alpha)): # = n_samples
                if self.alpha[j] > 0:
                    if self.kernel == 'gaussian':
                        prediction += self.alpha[j] * self.y[j] * gaussian_kernel(X[i], self.X[j], self.sigma) #devi vedere come chiamare il kernel
                    elif self.kernel == 'polynomial':
                        prediction += self.alpha[j] * self.y[j] * polynomial_kernel(X[i], X[j], self.c, self.d)
            predictions[i] = 1 if prediction >= 0 else -1

        return predictions , self.alpha

class KernelizedPegasos:
    def __init__(self,kernel='gaussian', learning_rate, max_iter=1000, sigma, c, d, lambda_par):
        self.kernel = kernel
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = None
        self.sigma = sigma
        self.c = c
        self.d = d
        self.lambda_par = lambda_par

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
                            prediction += self.alpha[j] * y[j] * gaussian_kernel(X[i], X[j], self.sigma)
                        elif self.kernel == 'polynomial':
                            prediction += self.alpha[j] * y[j] * polynomial_kernel(X[i], X[j], self.c, self.d)

                # Update alpha based on prediction
                if y[i] * prediction <= 1:  # Misclassified or on the margin
                    self.alpha[i] += self.learning_rate * (1 - y[i] * prediction)
                # Regularization step
                self.alpha[i] = max(0, self.alpha[i] - self.learning_rate * self.lambda_param * self.alpha[i]) # prevent overfitting and to ensure the weights don't grow too large. 
        # Identify support vectors
        self.support_vector_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = X[self.support_vector_indices]

'''Purpose: Stores the data points that are the support vectors.
Support Vectors: These are the data points that directly influence the decision boundary of the SVM. In a kernel SVM, support vectors are critical because
                 the decision boundary is defined by the points for which the Lagrange multipliers 𝛼  are non-zero.
Usage: During prediction, the kernel function is computed only between new data points and the support vectors. This helps make predictions more efficient 
        because you don't need to compute the kernel with all training data points, just the support vectors.'''
    
    def predict(self, X):
        predictions = []
        for x in X:
            prediction = 0
            for j in self.support_vector_indices:
                if self.kernel == 'gaussian':
                    prediction += self.alpha[j] * gaussian_kernel(x, self.support_vectors[j], self.sigma)  # Alpha for support vector
                elif self.kernel == 'polynomial':
                    prediction += self.alpha[j] * polynomial_kernel(x, self.support_vectors[j], self.c, self.d)
            predictions.append(np.sign(prediction))  # Sign of prediction
        return np.array(predictions)