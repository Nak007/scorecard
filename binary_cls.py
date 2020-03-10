'''
Functions
---------
(1) compare_classifers
\t Provides comparison among classifiers based on predefined 
\t metrics (scikit-learn interface)

(2) cls_n_features
\t Determines performance of predefined estimator for every 
\t increasing number of features

(3) train_test_plot
'''

import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as smt

def gini(y_true,y_proba):
    return 2*roc_auc_score(y_true,y_proba)-1

def compare_classifers(estimator, X, y, test_size=0.3, random_state=0, cutoff=0.5, 
                       metrics=[confusion_matrix, accuracy_score, gini]):

    '''
    ** Binary Classifier Comparison **
    This function provides comparison among classifiers based on 
    predefined metrics

    Parameters
    ----------
    estimator : list of classifer objects
    \t This is assumed to implement the scikit-learn estimator interface
    \t Methods    : self.fit(X, y), and self.predict_proba(X)
    \t Attributes : self.feature_importances_

    X : array-like of shape (n_samples, n_features)
    \t Dataset, where n_samples is the number of samples and n_features 
    \t is the number of features.

    y : array-like of shape (n_samples,)
    \t Binary target relative to X for classification or regression

    test_size : float, int or None, optional (default=0.3)
    \t It represents the proportion of the dataset to include in the test 
    \t split

    random_state : int, RandomState instance or None, optional (default=0)
    \t random_state is the random number generator used in split samples

    cutoff : float, optional (default=0.5)
    \t It must be between 0.0 and 1.0 and represents the probability 
    \t cutoff that defines targets and non-targets. This is used when 
    \t measuring metrics such as confusion_matrix or accuracy_score

    metrics : list of classification metrics,  
    optional (default=[confusion_matrix, accuracy_score, gini])
    \t This is assumed to implement the scikit-learn interface, where 
    \t instance must take inputs in the same manner e.g.
    \t accuracy_score(y_true, y_pred)

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
    >>> import cloudpickle
    >>> cloudpickle.dump(output.pkl,open('output.pkl','wb')) # save file
    >>> cloudpickle.load(open('output.pkl','rb')) # read file
    
    Example
    -------
    >>> from scorecard.binary_cls import compare_classifers
    >>> from sklearn import ensemble as em
    >>> from sklearn.metrics import f1_score, roc_auc_score
    
    >>> metrics = [f1_score, roc_auc_score] # Classifier Metrics
    >>> classifier = dict([('Random Forest', em.RandomForestClassifier())]
    >>> c = compare_classifers(classifier, X, y, metrics=metrics, cutoff=0.5)
    '''
    data, n_X, n_y = dict(), np.array(X), np.array(y)
    if isinstance(X,pd.core.frame.DataFrame): data['columns'] = list(X)
    else: data['columns'] = ['x'+str(n+1) for n in range(X.shape[1])]

    kwargs = dict(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = tts(n_X, n_y, **kwargs)
    data['data'] = dict([('train',{'X':X_train.tolist(),'y':y_train.tolist()}), 
                         ('test' ,{'X':X_test.tolist() ,'y':y_test.tolist()})])
    
    for _name_ in estimator.keys():
        start = time.time()
        print('Progress . . . Algorithm: {0}'.format(_name_))
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
        print('>>> Process Time : {:,.0f} seconds'.format(int(time.time()-start)))
    return data

def cls_n_features(classifier, X, y, n_feature=None, test_size=0.3, random_state=0, cutoff=0.5,
                   metrics=[confusion_matrix, accuracy_score, gini]):
    '''
    This function determines performance of predefined estimator for every 
    increasing number of features. The priority of features is stem from 
    their importance from initial fitting. Thus, it helps facilitate the
    decision in selecting the right number of features.

    Parameters
    ----------
    estimator : classifer object
    \t This is assumed to implement the scikit-learn estimator interface
    \t Methods    : self.fit(X, y), and self.predict_proba(X)
    \t Attributes : self.feature_importances_

    X : array-like of shape (n_samples, n_features)
    \t Dataset, where n_samples is the number of samples and n_features 
    \t is the number of features.

    y : array-like of shape (n_samples,)
    \t Binary target relative to X for classification or regression

    n_feature : int, optional (default=None)
    \t maximum number of features. If None, number of all features in X 
    \t is assigned.

    test_size : float, int or None, optional (default=0.3)
    \t It represents the proportion of the dataset to include in the test 
    \t split

    random_state : int, RandomState instance or None, optional (default=0)
    \t random_state is the random number generator and used in split 
    \t samples

    cutoff : float, optional (default=0.5)
    \t It must be between 0.0 and 1.0 and represents the probability cutoff
    \t that defines targets and non-targets. This is used when measuring
    \t metrics such as confusion_matrix or accuracy_score

    metrics : list of classification metrics,  
    optional (default=[confusion_matrix, accuracy_score, gini])
    \t This is assumed to implement the scikit-learn interface, where 
    \t instance must take inputs in the same manner e.g. 
    \t accuracy_score(y_true, y_pred)

    Return
    ------
    dictionary of *.pkl format
    {'columns': ['x20', 'x2', ...], # list of column names
     'index'  : [20, 2, ...], # list of column indices
     'test'   : {metric[0] : [...], metric[1] : [...], ...},
     'train'  : {metric[0] : [...], metric[1] : [...], ...}}

    ** Note **
    To write and read file
    >>> import cloudpickle
    >>> cloudpickle.dump(output.pkl,open('output.pkl','wb')) # save file
    >>> cloudpickle.load(open('output.pkl','rb')) # read file
    '''
    n_X, n_y = np.array(X), np.array(y)
    kwargs = dict(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = tts(n_X, n_y, **kwargs)
    classifier.fit(X_train,y_train)
   
    def _importance_(a):
        return a[1]
    if n_feature is None: n_feature = X.shape[1]
    n_var = [p for p in enumerate(classifier.feature_importances_)]
    n_var.sort(key=_importance_,reverse=True)
    
    if isinstance(X,pd.core.frame.DataFrame): columns = np.array(X.columns)
    else: columns = np.array(['x'+str(n+1) for n in range(X.shape[1])])
    data = dict([('train',dict([(n.__name__,[]) for n in metrics])),
                 ('test' ,dict([(n.__name__,[]) for n in metrics])),
                 ('model',[])])
    
    for m in range(n_feature):
        start = time.time()
        print('Feature : ({0}) {1}'.format(m+1, columns[n_var[m][0]]))
        index = [n_var[n][0] for n in range(m+1)]
        for (tp,ds,y_true) in zip(['train','test'],[X_train, X_test],[y_train, y_test]):
            classifier.fit(ds[:,index],y_true)
            y_proba = classifier.predict_proba(ds[:,index])[:,1]
            data['model'] += classifer
            for metric in metrics:
                try: retval = metric(y_true, y_proba)
                except: retval = metric(y_true, (y_proba>cutoff))
                if isinstance(retval,np.ndarray): retval = retval.reshape(-1).tolist()
                data[tp][metric.__name__] += [retval]
        print('>>> Process Time : {:,.0f} seconds'.format(int(time.time()-start)))
    data['index'] = index
    data['columns'] = columns[index].tolist()
    return data

def train_test_plot(axis, train, test, sigma=2, **kwargs):

    '''
    Plot train and test
    
    Parameters
    ----------
    axis : matplotlib axis object
    \t base class for axis in matplotlib

    train : array-like or 1D-array
    \t Training metric result

    test : array-like or 1D-array
    \t Testing metric result

    sigma : int
    \t Standard deviation for Gaussian kernel for
    \t scipy.ndimage.gaussian_filter1d().
    \t The higher the sigma, the smoother the curve

    **kwargs : keyword arguments
    \t Keyword arguments for "train_test_plot" function,
    \t (1) title : title of plot
    \t (2) ylabel : label of y-axis
    \t (3) xlabel : label of x-axis
    \t (4) xticklabels : label of x-axis ticks
    '''
    x = range(1,len(train)+1)
    axis.plot(x, smt(train,sigma=sigma), label='Train', lw=2, color='b')
    axis.plot(x, smt(test,sigma=sigma), label='Test', lw=2, color='r')

    if kwargs.get('title') is not None: 
        axis.set_title(kwargs.get('title'), fontsize=14)
    if kwargs.get('ylabel') is not None: 
        axis.set_ylabel(kwargs.get('ylabel'), fontsize=10)
    if kwargs.get('xlabel') is not None: 
        axis.set_xlabel(kwargs.get('xlabel'), fontsize=10)

    axis.set_xlim(1,max(x))
    axis.set_xticks(x)
    if kwargs.get('xticklabels') is not None:
        a = kwargs.get('xticklabels')
        axis.set_xticklabels(a,color='grey', rotation=90)
    kw = dict(loc=0,framealpha=0,facecolor=None,edgecolor=None)
    axis.legend(**kw)
