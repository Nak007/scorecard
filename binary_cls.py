'''
Instance
(1) compare_classifers
(2) cls_n_features
'''
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def gini(y_true,y_proba):
    return 2*roc_auc_score(y_true,y_proba)-1

def compare_classifers(estimator, X, y, test_size=0.3, random_state=0, cutoff=0.5, 
                       metrics=[confusion_matrix, accuracy_score, gini]):

    '''
    ** Binary Classifier Comparison **
    This function provides comparison among classifiers based on predefined metrics

    Parameters
    ----------

    estimator : list of classifer objects
    \t This is assumed to implement the scikit-learn estimator interface

    X : array-like of shape (n_samples, n_features)
    \t Dataset, where n_samples is the number of samples and n_features 
    \t is the number of features.

    y : array-like of shape (n_samples,)
    \t Binary target relative to X for classification or regression

    test_size : float, int or None, optional (default=0.3)
    \t It represents the proportion of the dataset to include in the test split

    random_state : int, RandomState instance or None, optional (default=0)
    \t random_state is the random number generator used in split samples

    cutoff : float, optional (default=0.5)
    \t It must be between 0.0 and 1.0 and represents the probability cutoff
    \t that defines targets and non-targets. This is used when measuring
    \t metrics such as confusion_matrix or accuracy_score

    metrics : list of classification metrics,  
    optional (default=[confusion_matrix, accuracy_score, gini])
    \t This is assumed to implement the scikit-learn interface, where instance
    \t must take inputs in the same manner i.e. accuracy_score(y_true, y_pred)

    Return
    ------

    dictionary of *.pkl format
    {'columns'  : ['x0','x1',...],
    'data'      : {'test'  : {'X' : [[...],[...]],'y' : [...]},
                   'train' : {'X' : [[...],[...]],'y' : [...]}},
    estimator_1 : {'importance : [...],
                   'model' : estimator,
                   'test'  : {metric[0] : m0, metric[1] : m1, ...},
                   'train' : {metric[0] : m0, metric[1] : m1, ...}},
    estimator_2 : { ... }}

    ** Note **
    To write and read file
    >>> import pickle
    >>> pickle.dump(output.pkl,open('output.pkl','wb')) # save file
    >>> pickle.load(open('output.pkl','rb')) # read file
    '''
    data = dict()
    if isinstance(X,pd.core.frame.DataFrame): data['columns'] = list(X)
    else: data['columns'] = ['x'+str(n+1) for n in range(X.shape[1])]
    
    X, y = np.array(X), np.array(y)
    kwargs = dict(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = tts(X, y, **kwargs)
    
    data['data'] = dict([('train',{'X':X_train.tolist(),'y':y_train.tolist()}), 
                         ('test' ,{'X':X_test.tolist() ,'y':y_test.tolist()})])
    
    for _name_ in estimator.keys():
        print('Progress . . . algorithm: {0}'.format(_name_))
        model = estimator[_name_]; model.fit(X_train, y_train)
        data[_name_] = dict([('model',model), ('train',None), ('test',None),
                             ('importance',model.feature_importances_.tolist())])
        for (tp,ds,y_true) in zip(['train','test'],[X_train, X_test],[y_train, y_test]):
            y_proba, r = model.predict_proba(ds)[:,1], [None] * len(metrics)
            for n,metric in enumerate(metrics):
                try: retval = metric(y_true, y_proba)
                except: retval = metric(y_true, (y_proba>cutoff))
                if isinstance(retval,np.ndarray): retval = retval.reshape(-1).tolist()
                r[n] = (metric.__name__,retval)
            data[_name_][tp] = dict(r)
    return data

def cls_n_features(classifier, X, y, n_feature=None, test_size=0.5, random_state=0, cutoff=0.5,
                   metrics=[confusion_matrix, accuracy_score, gini]):
    '''
    This function fits model for every increasing number of features based
    on their importance 

    Parameters
    ----------

    estimator : classifer object
    \t This is assumed to implement the scikit-learn estimator interface

    X : array-like of shape (n_samples, n_features)
    \t Dataset, where n_samples is the number of samples and n_features 
    \t is the number of features.

    y : array-like of shape (n_samples,)
    \t Binary target relative to X for classification or regression

    n_feature : int, optional (default=None)
    \t maximum number of features. If None, number of all features in X 
    \t is assigned.

    test_size : float, int or None, optional (default=0.3)
    \t It represents the proportion of the dataset to include in the test split

    random_state : int, RandomState instance or None, optional (default=0)
    \t random_state is the random number generator and used in split samples

    cutoff : float, optional (default=0.5)
    \t It must be between 0.0 and 1.0 and represents the probability cutoff
    \t that defines targets and non-targets. This is used when measuring
    \t metrics such as confusion_matrix or accuracy_score

    metrics : list of classification metrics,  
    optional (default=[confusion_matrix, accuracy_score, gini])
    \t This is assumed to implement the scikit-learn interface, where instance
    \t must take inputs in the same manner i.e. accuracy_score(y_true, y_pred)

    Return
    ------

    dictionary of *.pkl format
    {'index'  : [20, 2, ...], # column indices
     'test'   : {metric[0] : [...], metric[1] : [...], ...},
     'train'  : {metric[0] : [...], metric[1] : [...], ...}}

    ** Note **
    To write and read file
    >>> import pickle
    >>> pickle.dump(output.pkl,open('output.pkl','wb')) # save file
    >>> pickle.load(open('output.pkl','rb')) # read file
    '''
    kwargs = dict(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = tts(X, y, **kwargs)
    classifier.fit(X_train,y_train)
    a = dict([(n.__name__,[]) for n in metrics])
    data = dict([('train',a),('test',a)])

    def _importance_(a):
        return a[1]
    if n_feature is None: n_feature = X.shape[1]
    n_var = [p for p in enumerate(classifier.feature_importances_)]
    n_var.sort(key=_importance_,reverse=True)

    for n in range(n_feature):
        index = [n_var[n][0] for n in range(n+1)]
        for (tp,ds,y_true) in zip(['train','test'],[X_train, X_test],[y_train, y_test]):
            classifier.fit(ds[:,index],y_true)
            y_proba, r = classifier.predict_proba(ds[:,index])[:,1], [None]*len(metrics)
            for n,metric in enumerate(metrics):
                try: retval = metric(y_true, y_proba)
                except: retval = metric(y_true, (y_proba>cutoff))
                if isinstance(retval,np.ndarray): retval = retval.reshape(-1).tolist()
                data[tp][metric.__name__].append(retval)
    data['index'] = index
    return data
