'''
Instance:
(1) system_stability
(2) stability_index
(3) m_json
(4) prepare_xy
'''
import numpy as np, json, sys
from scipy.stats import chi2

def system_stability(X1, X2, bins, missing=0.05):
    
    '''
    ** System Stability Analysis **
    
    This analysis only indicates whether a shift has occured, and
    give an indication of the magnitude. The methods used in this 
    analysis are Population Stability Index (PSI) and Chi-Square.
    
    Note: 
    (1) missing is binned separately
    (2) variables are from those that exist in "X1", "X2", 
        and "bins"
    (3) size of X1 can be different from X2
    
    Parameters
    ----------
    X1 : dataframe, shape of (n_sample, n_feature)
    \t a reference dataset e.g. develpment sample
    
    X2 : dataframe, list of datafarmes, 
         shape of (n_sample, n_feature)
    \t a compared dataset, where shift is determined
    
    bins : dictionary of bin edges
    \t i.e. {'var_1':[-inf,0,1,2,inf],'var_2':[-inf,0,5,8,inf]}
    
    missing : float, optional, default: 0.05 (5%)
    \t if percent distribution is missing, default value is used
    \t when calculate Population Stability Index (PSI)
    
    Returns
    -------
    dictionary of results derived from stability_index (method).
    Indices are arranged in accordance with items in X2-list.
    
    Example
    -------
    >>> a  = np.random.randint(0, 5, size=(100, 2)
    >>> X1 = pd.DataFrame(a, columns=list('AB'))
    >>> b  = np.random.randint(0, 5, size=( 50, 2))
    >>> X2 = pd.DataFrame(b, columns=list('AB'))
    >>> bins = dict((n,np.arange(0,6)) for n in ['A','B'])
    >>> bins
    {'A': array([0, 1, 2, 3, 4, 5]), 
     'B': array([0, 1, 2, 3, 4, 5])}
    >>> system_stability(X1, X2, bins)
    {'_feature_': ['B', 'A'],
     'A' : {'columns': ['lower','upper','p_actual','p_expect','n_psi'],
     'data': [(n_bins, 5)], 'psi' : float, 
     'crit_val': float, 'p_value' : float},
     'B' : …}
  
    Note
    ----
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
            
            PSI = ∑{(%A(i)-%E(i))*LOG(%A(i)/%E(i))}, i ∈ 1,2,…,n
        
        In addition, %A(i) and %E(i) can be expressed as:
        
                      %A(i) = A(i)/A, %E(i) = E(i)/E
          
        where "A(i)" and "E(i)" are actual and expected amount of 
        ith bin, and "n" is a number of bins.
    
    (2) Using Chi-Square to test Goodness-of-Fit-Test (χ)
        The goodness of fit test is used to test if sample data fits 
        a distribution from a certain population. Its formula is 
        expressed as:
        
                  χ = ∑{(O(i)-E(i))^2/E(i)}, i ∈ 1,2,…,n
                
        where O(i) and E(i) are observed and expected percentages of 
        ith bin, and n is a number of bins
    
    Parameters
    ----------
    x1 : 1D-array, shape of (n_sample_1)
    \t array of actual or observed values
    
    x2 : 1D-array, shape of (n_sample_2)
    \t array of expected values
    \t Note: size of x1 can be different from x2
    
    bins : int or sequence of scalars, optional, (default:10)
    \t If bins is an int, it defines the number of equal-width  
    \t bins in the given range. If bins is a sequence, it defines 
    \t a monotonically  increasing array of bin edges, including 
    \t the rightmost edge, allowing for non-uniform bin widths. 
    \t Frequency in each bin is defined as bins[i] <= x < bins[i+1]
    \t **Note**
    \t missing or np.nan will be binned separately
    
    missing : float, optional, default: 0.05 (5%)
    \t if percent distribution is missing, default value is used
    \t when calculate Population Stability Index (PSI)
    
    Returns
    -------
    dictionary of (*.json)
    (1) "columns" : shape of (5,)
    (2) "data" : shape of ('missing' + n_bin, 5)
    (3) "psi" : float (Population Stability Index, PSI)
    (4) "crit_val" : float (critical value for Chi-Square)
    (5) "p_value" : float (cdf given critical value)
    
    Example
    -------
    >>> from scorecard.monitoring import *
    >>> import np as numpy, pandas as pd
    
    >>> a = np.random.randint(0,5,100)
    >>> b = np.random.randint(0,5, 50)
    >>> inf = float('Inf')
    >>> bins = [-inf, 1, 2, 3, 4, inf]
    >>> c = stability_index(a, b, bins=bins)
    >>> c
    {'columns' : ['lower', 'upper', 'p_actual', 'p_expect', 'n_psi'],
     'data'    : [[ nan, nan, 0.05, 0.05, 0.0  ],
                  [-inf, 1.0, 0.22, 0.18, 0.008],
                  [ 1.0, 2.0, 0.19, 0.22, 0.004],
                  [ 2.0, 3.0, 0.21, 0.14, 0.028],
                  [ 3.0, 4.0, 0.21, 0.26, 0.011],
                  [ 4.0, inf, 0.17, 0.2 , 0.005]],
     'psi'     : 0.05636,
     'crit_val': 0.06210,
     'p_value' : 0.80321}
    
    Note
    ----
    In order to view nested dictionaries, use the following code
    >>> import pprint 
    >>> pprint.PrettyPrinter(indent=1).pprint(json_file)
    '''
    x, p, si = x1.tolist() + x2.tolist(), [None,None], dict()
    if isinstance(bins,int): 
        bins = np.histogram(x, bins=bins)[1].tolist()
    else: bins = list(bins)
        
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
    si['crit_val'] = (np.diff(np.hstack(p),axis=1)**2/exp).sum()
    si['p_value'] = 1-chi2.cdf(si['crit_val'], df=1)
    return si

def m_json(file, data=None, mode='r', indent=None, encoding='utf-8'):
    
    '''
    Read and write JSON file
    
    Parameters
    ----------
    file : path-like object 
    \t Object represents a file system path e.g. sample.json
    
    data : serialize obj as a JSON, optional, (default:None)
    \t data is required only when mode is 'w'. 
    
    mode : str, optional, default: 'r' 
    \t mode while opening a file. If not provided, it defaults  
    \t to 'r'. Available file modes are 'r' (read), and 'w' 
    \t (write)
    
    indent : int, optional, (default:None)
    \t indent is applied when mode is 'w'
    
    encoding : str, optional, (default:'utf-8')
    \t encoding to use for UTF when reading / writing 
    \t (cp874:Thai)
    
    Reference
    ---------
    (1) https://docs.python.org/2/library/json.html
    (2) https://www.programiz.com/python-programming/methods/
        built-in/open
        
    Example
    -------
    # Write JSON file
    >>> m_json('sample.json', data=sample, mode='w')
    Successfully saved ==> sample.json
    
    # Read JSON file
    >>> h = m_json('sample.json', mode='r')
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
