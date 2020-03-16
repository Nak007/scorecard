'''
Functions
---------
(1) compare_classifers
(2) cls_n_features
(3) train_test_plot
(4) confustion_matrix_test
(5) sort_features
'''

import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as smt
import ipywidgets as widgets
from IPython.display import display

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

    Returns
    -------
    dictionary of *.pkl format
    {'data'        : {'test'  : {'X' : [[...],[...]],'y' : [...]},
                      'train' : {'X' : [[...],[...]],'y' : [...]}},
     'estimator_1' : {'importance : [...],
                      'model' : estimator,
                      'test'  : {metric[0] : m0, metric[1] : m1, ...},
                      'train' : {metric[0] : m0, metric[1] : m1, ...}},
     'estimator_2' : { ... }}

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
    data = dict()
    if isinstance(X,pd.core.frame.DataFrame): columns = list(X)
    else: columns = ['x'+str(n+1) for n in range(X.shape[1])]
    
    kwargs = dict(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = tts(np.array(X), np.array(y), **kwargs)
    a = pd.DataFrame(X_train,columns=columns).to_dict(orient='list')
    b = pd.DataFrame(X_test, columns=columns).to_dict(orient='list')
    data['data'] = dict([('train',{'X':a,'y':y_train.tolist()}), 
                         ('test' ,{'X':b,'y':y_test.tolist()})])
    
    t, f = progress_bar()
    for c,_name_ in enumerate(estimator.keys(),1):
        start = time.time()
        t.value = 'Progress . . . Algorithm: {0}'.format(_name_)
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
        f.value = c/len(estimator)*100
        #print('Process Time : {:,.0f} seconds'.format(int(time.time()-start)))
    f.bar_style='success'; t.value = 'Complete . . .'
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

    Returns
    -------
    dictionary of *.pkl format
    {'data'   : {'test'  : {'X' : [[...],[...]],'y' : [...]},
                 'train' : {'X' : [[...],[...]],'y' : [...]}},
     'columns': ['x20', 'x2', ...], # list of column names
     'index'  : [20, 2, ...], # list of column indices
     'test'   : {metric[0] : [...], metric[1] : [...], ...},
     'train'  : {metric[0] : [...], metric[1] : [...], ...}}
                  
    ** Note **
    To write and read file
    >>> import cloudpickle
    >>> cloudpickle.dump(output.pkl,open('output.pkl','wb')) # save file
    >>> cloudpickle.load(open('output.pkl','rb')) # read file
    '''
    t, f = progress_bar()
    kwargs = dict(test_size=test_size, random_state=random_state)
    t.value = 'Fitting model . . .'
    X_train, X_test, y_train, y_test = tts(np.array(X), np.array(y), **kwargs)
    a = pd.DataFrame(X_train,columns=columns).to_dict(orient='list')
    b = pd.DataFrame(X_test, columns=columns).to_dict(orient='list')
    data['data'] = dict([('train',{'X': a,'y': y_train.tolist()}), 
                         ('test' ,{'X': b,'y': y_test.tolist()})])
    classifier.fit(X_train,y_train)
   
    def _importance_(a):
        return a[1]
    if n_feature is None: n_feature = X.shape[1]
    n_var = [p for p in enumerate(classifier.feature_importances_)]
    n_var.sort(key=_importance_,reverse=True)
    
    if isinstance(X,pd.core.frame.DataFrame): columns = np.array(X.columns)
    else: columns = np.array(['x'+str(n+1) for n in range(X.shape[1])])
    data = dict([('train',dict([(n.__name__,[]) for n in metrics])),
                 ('test' ,dict([(n.__name__,[]) for n in metrics]))])
    
    for m in range(n_feature):
        start = time.time()
        t.value = 'Feature : ({0}) {1}'.format(m+1, columns[n_var[m][0]])
        index = [n_var[n][0] for n in range(m+1)]
        for (tp,ds,y_true) in zip(['train','test'],[X_train, X_test],[y_train, y_test]):
            classifier.fit(ds[:,index],y_true)
            y_proba = classifier.predict_proba(ds[:,index])[:,1]
            for metric in metrics:
                try: retval = metric(y_true, y_proba)
                except: retval = metric(y_true, (y_proba>cutoff))
                if isinstance(retval,np.ndarray): retval = retval.reshape(-1).tolist()
                data[tp][metric.__name__] += [retval]
        f.value = (m+1)/n_feature*100
        #print('>>> Process Time : {:,.0f} seconds'.format(int(time.time()-start)))
    data['index'] = index
    data['columns'] = columns[index].tolist()
    f.bar_style='success'; t.value = 'Complete . . .'
    return data

def train_test_plot(axis, train, test, **kwargs):

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

    **kwargs : keyword arguments
    \t Keyword arguments for "train_test_plot" function,
    \t  (1) title  : the title of plot (default:None)
    \t  (2) ylabel : the label text of y-axis (default:None)
    \t  (3) xlabel : the label text of x-axis (default:None)
    \t  (4) labels : the x-tick labels with list of string
    \t      labels (default:None)
    \t  (5) linewidth : the line width in points (default:1)
    \t  (6) linestyle : the linestyle of the line 
    \t      (default:'--')
    \t  (7) colors : tuple of color-hex codes = (train,test) 
    \t      (default:('b','r'))
    \t  (8) sigma : int, Standard deviation for Gaussian 
    \t      kernel for scipy.ndimage.gaussian_filter1d().
    \t      The higher the sigma, the smoother the curve
    \t  (9) decimal : number of decimal places for y
    \t (10) plot : string of plot name i.e. 'step' or 'line'
    '''
    x = range(1,len(train)+1)
    k = dict(decimal=2, sigma=2, tiltle=None, ylabel=None, xlabel=None, linewidth=2, 
             linestyle='-', labels=None, colors=('b','r'), plot='step')
    k = {**k,**kwargs}

    for (n,d) in enumerate(zip([train,test],['Train','Test'])):
        c = dict(lw=k['linewidth'], ls=k['linestyle'], label=d[1], color=k['colors'][n])
        if k['sigma'] > 0: y = smt(d[0],k['sigma'])
        else: y = d[0]
        if k['plot']=='step': axis.step(x, np.round_(y,k['decimal']) , **c)
        else: axis.plot(x, np.round_(y,k['decimal']) , **c)
  
    if k['title'] is not None: 
        axis.set_title(k['title'], fontsize=14)
    if k['ylabel'] is not None: 
        axis.set_ylabel(k['ylabel'], fontsize=10)
    if k['xlabel'] is not None: 
        axis.set_xlabel(k['xlabel'], fontsize=10)
    axis.set_xlim(0.5,max(x)+0.5)
    axis.set_xticks(x)
    if k['labels'] is not None:
        axis.set_xticklabels(k['labels'],color='grey', rotation=90)
    kw = dict(loc=0,framealpha=0,facecolor=None,edgecolor=None)
    axis.legend(**kw)
    
def confustion_matrix_test(train, test, figsize=(15,10), fname=None, **kwargs):

    '''
    Description
    -----------
    - True Positive Rate (TPR): when the actual is positive, 
      how often does the model actually predict it correctly
    - False Positive Rate (FPR): probability of false alarm
    - Precision: when the prediction is positive, how often 
      is it correct
    - Negative Predictive Value (NPV): when the prediction 
      is negative, how often is it correct
    - Positive Likelihood Ratio: odds of a positive 
      prediction given that the person is sick
    
    Parameters
    ----------
    train : (n_sample,4)
    \t Results from training dataset ['TN','FP','FN','TP']

    test : (n_sample,4)
    \t Results from testing dataset ['TN','FP','FN','TP']
    
    figsize : (float, float), optional, default: (15,10)
    \t width, height in inches
    
    fname : str or PathLike or file-like object
    \t A path, or a Python file-like object 
    \t see pyplot.savefig() for more information
    
    **kwargs : keyword arguments
    \t Keyword arguments for "train_test_plot()"
    '''
    fig = plt.figure(figsize=figsize)
    shape = (3,3)

    # Confusion Matrix output = ['TN','FP','FN','TP']
    tp = plt.subplot2grid(shape, (0, 0))
    fn = plt.subplot2grid(shape, (1, 0))
    fp = plt.subplot2grid(shape, (0, 1))
    tn = plt.subplot2grid(shape, (1, 1))

    # metrics
    tpr = plt.subplot2grid(shape, (2, 0))
    fpr = plt.subplot2grid(shape, (2, 1))
    # Positive predictive value (PPV), Precision
    ppv = plt.subplot2grid(shape, (0, 2))
    # Negative predictive value (NPV) 
    npv = plt.subplot2grid(shape, (1, 2))
    # Positive likelihood ratio (LR+)
    lr = plt.subplot2grid(shape, (2, 2))

    # trian and test results
    train = train/np.sum(train,axis=1)[0]*100
    test = test/np.sum(test,axis=1)[0]*100

    def bold_font(s):
        return ' '.join([r'$\bf{0}$'.format(n) for n in s.split(' ')])

    g = {**kwargs,**dict(ylabel='Percent (%)')}
    t = bold_font('True Positive (TP)')
    train_test_plot(tp,train[:,3], test[:,3], **{**g, **dict(title=t)})
    t = bold_font('False Negative (FN)') + '\n(Type II error)'
    train_test_plot(fn,train[:,2], test[:,2], **{**g, **dict(title=t)})
    t = bold_font('False Positive (FP)') + '\n(Type I error)'
    train_test_plot(fp,train[:,1], test[:,1], **{**g, **dict(title=t)})
    t = bold_font('True Negative (TN)')
    train_test_plot(tn,train[:,0], test[:,0], **{**g, **dict(title=t)})

    t = bold_font('True Positive Rate (TPR)') + '\n TP / (TP + FN) '
    tpr1 = train[:,3]/train[:,[2,3]].sum(axis=1)*100
    tpr2 = test[:,3]/test[:,[2,3]].sum(axis=1)*100
    train_test_plot(tpr ,tpr1, tpr2, **{**g, **dict(title=t)})
    tpr.set_facecolor('#f1f2f6')

    t = bold_font('False Positive Rate (FPR)') + '\n FP / (FP + TN) '
    fpr1 = train[:,1]/train[:,[0,1]].sum(axis=1)*100
    fpr2 = test[:,1]/test[:,[0,1]].sum(axis=1)*100
    train_test_plot(fpr ,fpr1, fpr2, **{**g, **dict(title=t)})
    fpr.set_facecolor('#f1f2f6')

    t = bold_font('Precision') + '\n TP / (TP + FP) '
    p1 = train[:,3]/train[:,[1,3]].sum(axis=1)*100
    p2 = test[:,3]/test[:,[1,3]].sum(axis=1)*100
    train_test_plot(ppv , p1, p2, **{**g, **dict(title=t)})
    ppv.set_facecolor('#f1f2f6')

    t = bold_font('Negative Predictive Rate (NPR)') + '\n TN / (TN + FN) '
    npr1 = train[:,0]/train[:,[0,2]].sum(axis=1)*100
    npr2 = test[:,0]/test[:,[0,2]].sum(axis=1)*100
    train_test_plot(npv , npr1, npr2, **{**g, **dict(title=t)})
    npv.set_facecolor('#f1f2f6')

    t = bold_font('Positive likelihood ratio') + '\n TPR / FPR '
    lr1, lr2 = tpr1/fpr1, tpr2/fpr2
    train_test_plot(lr ,lr1, lr2, **{**g, **dict(title=t)})
    lr.set_facecolor('#f1f2f6')

    fig.tight_layout()
    if fname != None: plt.savefig(fname)
    plt.show()
    
def progress_bar():
    
    kwargs = dict(value=0, min=0, max=100, step=0.01, 
                  bar_style='info', orientation='horizontal')
    f_text = widgets.HTMLMath(value='Calculating...')
    f_prog = widgets.FloatProgress(**kwargs)
    w = widgets.VBox([f_text, f_prog])
    display(w); time.sleep(2)
    return f_text, f_prog

def sort_features(features, importances):
    
    '''
    Return a sorted copy of an array.
    
    Parameters
    ----------
    features : list of str
    \t An array of column names
    
    importances : list of floats
    \t An array of importances, of the same shape as features.
    '''
    def _importance_(a):
        return a[1]
    n_var = [p for p in zip(features, importances)]
    n_var.sort(key=_importance_,reverse=True)
    return n_var
