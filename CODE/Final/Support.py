import numpy as np  # type: ignore
import pandas as pd # type: ignore
import itertools as it

def Outliers(X, Y):
    '''Removes outliers using the Interquartile Range (IQR) method.'''
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X_cleaned = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    Y_cleaned=Y[X_cleaned.index]
    return X_cleaned, Y_cleaned

def Split(X, Y):
    '''Splits the dataset into 70% training and 30% testing.'''
    train = X.sample(frac=0.7, random_state=42)  
    test = X.drop(train.index)
    X_test=test.values
    X_train=train.values
    Y_train=Y.loc[train.index].values
    Y_test=Y.loc[test.index].values
    return X_train, X_test, Y_train, Y_test

def Standardized(X_train, X_test):
    '''Standardizes features using mean and standard deviation of the training set.'''
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    X_train_standardized = (X_train - train_mean) / train_std
    X_test_standardized = (X_test - train_mean) / train_std
    return X_train_standardized, X_test_standardized

def Zero_One_Loss(Y_true, Y_hat):
    '''Computes the zero-one loss (misclassification rate).'''
    numbermiss = np.mean(Y_hat != Y_true)
    return numbermiss


def Polynomial_exp(X_train,X_test, column_names): 
    '''Expands features with quadratic terms and pairwise interactions.'''
    X_train_df=pd.DataFrame(X_train, columns=column_names)
    X_test_df=pd.DataFrame(X_test, columns=column_names)
    # add squared terms
    for i in column_names:
        X_train_df[f'{i}^2']=X_train_df[i]**2
        X_test_df[f'{i}^2']=X_test_df[i]**2
    # add interaction terms
    for i in range(len(column_names)):
        for j in range(i+1,len(column_names)):
            col1 = column_names[i]
            col2 = column_names[j]
            X_train_df[f'{col1}*{col2}'] = X_train_df[col1] * X_train_df[col2]
            X_test_df[f'{col1}*{col2}'] = X_test_df[col1] * X_test_df[col2]

    return X_train_df , X_test_df

def k_Cross_Validation(model_class, X, y, params, k):
    """    
    Performs k-fold cross-validation and returns the average loss.
    model_class: the class of the model (Perceptron, Pegasos, etc.)
    X: DataFrame of the features
    y: Array of the outcomes
    params: parameters
    k: fold number"""
    np.random.seed()
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  

    scores= []

    for i in range(k):
        start = i*fold_size
        if i < k-1 :
            end = start + fold_size
        else :
            end = n_samples
        
        validation_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, validation_idx)
        # validaation and training set
        X_train, X_val = X.iloc[train_idx], X.iloc[validation_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[validation_idx]
        
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        y_hat = model.predict(X_val)
        loss =  Zero_One_Loss (y_val, y_hat)

        scores.append(loss)
    
    return np.mean(scores)


def tuning_par(model, X, y, params, k): 
    '''
    model_class: the class of the model (Perceptron, Pegasos, etc.)
    X: DataFrame of the features
    y: Array of the outcomes
    param_grid: dictionary with parameters (es. {'eta': [0.01, 0.1], 'n_iter': [100, 200]})
    k: number of folds for cross-validation
    
    output: best set of parameters and the loss.
    '''
    
    best_param = None
    best_loss=np.inf

    keys, values = zip(*params.items())
    param_combinations = [dict(zip(keys, v)) for v in it.product(*values)]

    for i in param_combinations:
        loss = k_Cross_Validation(model, X, y, i, k=k)
        if(loss<best_loss):
            best_loss=loss
            best_param=i
    
    return best_loss, best_param

def weights_comparison(w1, w2, w3, column_names):
    '''Compares feature importance across Perceptron, Pegasos, and Logistic Regression.'''
    weights_df = pd.DataFrame({
    'Feature': column_names,
    'Perceptron': np.abs(w1)/np.abs(w1).sum(), #relative importance of the weights on the model as a percentage
    'Pegasos': np.abs(w2)/np.abs(w2).sum(),
    'Logistic_Regression': np.abs(w3)/np.abs(w3).sum()
    }
    )
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', None)  
    pd.set_option('display.max_colwidth', None)  

    print(weights_df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

def gaussian_kernel_vectorized(X1, X2, sigma=1.0):
    """
    Parameters:
    X1: The first dataset with n1 data points, each with d features.
    X2: The second dataset with n2 data points, each with d features.
    sigma: The standard deviation (spread) of the Gaussian kernel.
    Returns:
    kernel_matrix: ndarray of shape (n1, n2)
        A matrix where the (i, j)-th entry represents the kernel value between X1[i] and X2[j].
    """
    # Compute the squared norm of each row in X1 and X2
    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)  # Shape: (n1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)  # Shape: (1, n2)

    # Compute the squared Euclidean distance between each pair of points
    dists = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)  # Shape: (n1, n2)

    # Apply the Gaussian kernel formula
    kernel_matrix = np.exp(-dists / (2 * sigma ** 2))

    return kernel_matrix

def polynomial_kernel_vectorized(X, Z, c=1.0, d=2):
    """
    Parameters:
    X:First dataset with n1 data points, each with d features.
    Z:Second dataset with n2 data points, each with d features.
    c:Constant term added to the dot product in the polynomial kernel.
    d:Degree of the polynomial.
    Returns:
    kernel_matrix: ndarray of shape (n1, n2)
        A matrix where the (i, j)-th entry represents the polynomial kernel value between X[i] and Z[j].
    """
    return (np.dot(X, Z.T) + c) ** d 
