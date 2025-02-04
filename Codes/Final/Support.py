import numpy as np 
import pandas as pd
import itertools as it

def Outliers(X, Y):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    # Filtra i dati rimuovendo gli outlier
    X_cleaned = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    Y_cleaned=Y[X_cleaned.index]
    return X_cleaned, Y_cleaned

def Split(X, Y):
    train = X.sample(frac=0.7, random_state=42)  # Estrae il 70% dei dati in modo casuale
    test = X.drop(train.index)
    X_test=test.values
    X_train=train.values
    Y_train=Y.loc[train.index].values
    Y_test=Y.loc[test.index].values
    return X_train, X_test, Y_train, Y_test

def Standardized(X_train, X_test):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    # Standardizza il training set
    X_train_standardized = (X_train - train_mean) / train_std
    # Usa la stessa media e std per standardizzare il test set
    X_test_standardized = (X_test - train_mean) / train_std
    return X_train_standardized, X_test_standardized

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
        loss =  Zero_One_Loss (y_val, y_hat)

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

def weights_comparison(w1, w2, w3, column_names):
    weights_df = pd.DataFrame({
    'Feature': column_names,
    'Perceptron': np.abs(w1)/np.abs(w1).sum(), #importanza relativa dei pesi sul modello come percentuale
    'Pegasos': np.abs(w2)/np.abs(w2).sum(),
    'Logistic_Regression': np.abs(w3)/np.abs(w3).sum()
    }
    )
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', None)  
    pd.set_option('display.max_colwidth', None)  # Mostra interamente il contenuto delle celle

    print(weights_df)
    # Ripristina le opzioni dopo la visualizzazione se desideri tornare alla modalità predefinita
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

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
