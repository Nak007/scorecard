'''
Instance:
(1) system_stability
(2) stability_index
(3) m_json
(4) prepare_xy
'''
import numpy as np, json, matplotlib.pyplot as plt, sys
from scipy.stats import chi2

def system_stability(X1, X2, bins, missing=0.05):
    
    '''
    ** System Stability Analysis **
    
    This analysis only indicates whether a shift has occured, and
    give an indication of the magnitude. The methods used in this 
    analysis are Population Stability Index (PSI) and Chi-Square.
    
    Note: 
    (1) missing is binned separately
    (2) variables are from those that exist in "X1", "X2", and "bins"
    (3) size of X1 can be different from X2
    
    Parameters
    ----------
    
    X1 : dataframe, shape of (n_sample, n_feature)
    \t a reference dataset e.g. develpment sample
    
    X2 : dataframe, list of datafrmes, shape of (n_sample, n_feature)
    \t a compared dataset, where shift is determined
    
    bins : dictionary of bin edges
    \t i.e. {'var_1':[-inf,0,1,2,inf], 'var_2':[-inf,0,5,8,inf]}
    
    missing : float, optional, default: 0.05 (5%)
    \t if percent distribution is missing, default value is used when
    \t calculate Population Stability Index (PSI)
    
    Return
    ------
    
    dictionary of results derived from stability_index (instance).
    Indices are arranged in accordance with items in X2-list.
    
    ** Note **
    In order to view nested dictionaries, use the following code
    >>> import pprint 
    >>> pprint.PrettyPrinter(indent=1).pprint(json_file)
    '''
    if isinstance(X2,list)==False: X2 = [X2]
    features, ssa = set(bins).intersection(X1.columns), dict()
    for (n,X) in enumerate(X2):
        n_features, a = set(X.columns).intersection(features), dict()
        if bool(n_features):
            a['_feature_'] = list(n_features)
            for var in n_features:
                a[var] = stability_index(X1[var],X[var],bins[var],missing)
        ssa[n] = a
    if len(X2)==1: return ssa[0]
    else: return ssa
    
def stability_index(x1, x2, bins=10, missing=0.05):
    
    '''
    This function only takes binary classification i.e. 0 and 1,
    where 0 indicates non-event whereas 1 indicates event (target).
    
    (1) Population Stability Index (PSI)
    
    =================================================================
    |  PSI Value  |      Inference         |        Action          |
    -----------------------------------------------------------------
    |    < 0.10   | no significant change  | no action required     |
    |  0.1 – 0.25 | small change           | investigation required |       
    |    > 0.25   | Major shift            | need to delve deeper   |
    =================================================================
    PSI = sum((%Actual-%Expect)*LOG(%Actual/%Expect))
    
    (2) Using Chi-Square to test Goodness-of-Fit-Test
    The goodness of fit test is used to test if sample data fits 
    a distribution from a certain population. Its formula is expressed
    as X2 = sum((O-E)**2/E)
    Null Hypothesis: two sampels are fit the expected population
    
    Parameters
    ----------
    
    x1 : 1D-array, shape of (n_sample_1)
    \t array of actual or observed values
    
    x2 : 1D-array, shape of (n_sample_2)
    \t array of expected values
    \t Note: size of x1 can be different from x2
    
    bins : int or sequence of scalars, optional, default: 10
    \t If bins is an int, it defines the number of equal-width bins in 
    \t the given range. If bins is a sequence, it defines 
    \t a monotonically  increasing array of bin edges, including the 
    \t rightmost edge, allowing for non-uniform bin widths. 
    \t Frequency in each bin is defined as bins[i] <= x < bins[i+1]
    \t Note: missing or np.nan will be binned separately
    
    missing : float, optional, default: 0.05 (5%)
    \t if percent distribution is missing, default value is used when
    \t calculate Population Stability Index (PSI)
    
    Return
    ------
    
    dictionary of (*.json)
    (1) "columns" : shape of (5,)
    (2) "data" : shape of ('missing' + n_bin, 5)
    (3) "psi" : float (Population Stability Index, PSI)
    (4) "crit_val" : float (critical value for Chi-Square)
    (5) "p_value" : float (cdf given critical value)
    
    ** Note **
    In order to view nested dictionaries, use the following code
    >>> import pprint 
    >>> pprint.PrettyPrinter(indent=1).pprint(json_file)
    '''
    x, p, si = x1.tolist() + x2.tolist(), [None,None], dict()
    if isinstance(bins,int): bins = np.histogram(x, bins=bins)[1].tolist()
    for n,x in enumerate([x1,x2]):
        freq = np.histogram(x[~np.isnan(x)], bins=bins)[0]
        freq = [np.isnan(x).sum()] + freq.tolist()
        p[n] = (np.array(freq)/sum(freq)).reshape(-1,1)
        p[n][(p[n]==0) | (np.isnan(p[n]))] = missing
    
    # Population Stability Index
    n_psi = ((p[0]-p[1])*np.log(p[0]/p[1])).reshape(-1,1)
    n_psi[np.isnan(n_psi)] = 0
    si['columns'] = ['lower', 'upper', 'p_actual', 'p_expect', 'n_psi']
    lower = np.array([np.nan] + bins[:-1]).reshape(-1,1)
    upper = np.array([np.nan] + bins[1:]).reshape(-1,1)
    si['data'] = np.hstack((lower,upper,p[0],p[1],n_psi)).tolist()
    si['psi'] = n_psi.sum(axis=None)
    
    # Chi-Square Test
    exp = np.where(p[1]==0,1,p[1])
    si['crit_val'] = (np.diff(np.hstack(p),axis=1)**2/exp).sum(axis=None)
    si['p_value'] = 1-chi2.cdf(si['crit_val'], df=1)
    return si

def m_json(file, data=None, mode='r', indent=None, encoding='utf-8'):
    
    '''    
    Parameters
    ----------
    
    file : path-like object (representing a file system path)
    
    data : serialize obj as a JSON, optional, default: None
    \t data is required only when mode is 'w'. 
    
    mode : str, optional, default: 'r' 
    \t mode while opening a file. If not provided, it defaults to 'r' 
    \t Available file modes are 'r' (read), and 'w' (write)
    
    indent : int, optional, default: None
    \t indent is applied when mode is 'w'
    
    encoding : str, optional, default: 'utf-8'
    \t encoding to use for UTF when reading/writing (cp874: Thai)
    
    Reference
    ---------
    (1) https://docs.python.org/2/library/json.html
    (2) https://www.programiz.com/python-programming/methods/built-in/open
    '''
    if (mode=='w') & (data is not None):
        try:
            with open(file, mode, encoding=encoding) as f:
                json.dump(data,f,indent=indent)
            print('Successfully saved ==> {0}'.format(file))
        except: print("Unexpected error: {0}".format(sys.exc_info()[0]))
    elif mode=='r':
        try:
            with open(file, mode, encoding=encoding) as f:
                return json.load(f)
        except: print("Unexpected error: {0}".format(sys.exc_info()[0]))
            
def prepare_xy(a, dict_keys=None, features=None, metrics=['p_value','psi']):
  
    '''
    Parameters
    ----------

    a : dictionary from *.JSON file from [system_stability]

    dict_keys : list or dict_keys, optional, default: None
    \t if dict_keys is not defined (None), number of keys e.g. 0,1,..n
    \t is automatically determined from the input file (a)

    features : list of features, optional, default: None
    \t if list of features is not defined (None), number of features is
    \t automatically determined from the input file (a)

    metrics : list of measurements, optional, default: ['p_value','psi']
    \t list of measurements that is used to quantify the distribution
    \t shift i.e. chi-square (p_value) and Population Stability Index (psi)

    Return
    ------

    dictionary, where dict_keys = metrics

    ** Note **
    In order to view nested dictionaries, use the following code
    >>> import pprint 
    >>> pprint.PrettyPrinter(indent=1).pprint(json_file)
    '''
    f_data = dict()
    for metric in metrics:
        data, f_data[metric] = dict(), None
        if dict_keys is None: dict_keys = a.keys()
        for key in dict_keys:
            if features is None: n_feature = a[str(key)]['_feature_']
            else: n_feature = features.copy()
            for f in n_feature:
                if data.get(f)==None: data[f] = dict([('x',[]),('y',[])])
                data[f]['x'] += [int(key)]
                data[f]['y'] += [a[str(key)][f][metric]]
        f_data[metric] = data
    return f_data
