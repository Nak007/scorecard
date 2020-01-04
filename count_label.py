import numpy as np, pandas as pd

def consec_list(a, c=1, key=None):

    '''
    Parameters
    ----------

    a : list of shape (n_position)
    \t list must contain only integer labels

    c : integer, optional, default: 1
    \t number of labels appears consecutively

    key : list, optional, default: None
    \t list of key labels. 
    \t If None, unique labels are determined from the list

    Return
    ------

    list that contains numnber of occurances for each label with 
    respect to consecutiveness as well as key labels
    '''
    if key is None: key = np.unique(a).tolist()
    if (c<=len(a)) & (c>0):
        b = [min(a)-1] + a + [min(a)-1]
        b = np.array([b[n:n+(c+2)] for n in np.arange(0,len(b)-c-1)])
        return [sum((b[:,0]!=n) & (b[:,-1]!=n) & 
                    ((b[:,1:-1]==n).sum(axis=1)==c)) for n in key], key
    else: return [0] * len(key), key

def consec_array(a, c=1, key=None):

    '''
    Parameters
    ----------

    a : array-like of shape (n_samples, n_columns)
    \t 2D-array must contain only integer labels

    c : integer, optional, default: 1
    \t number of labels appears consecutively

    key : list of integers, optional, default: None
    \t list of key labels. 
    \t If None, unique labels are determined from the array

    Return
    ------

    array of shape (n_samples, n_labels) that contains numnber of occurances 
    for each label with respect to consecutiveness as well as key labels
    '''
    if key is None: key = np.unique(a).tolist()
    if (c<=a.shape[1]) & (c>0):
        n_min = np.full((len(a),1),a.min()-1)
        b = np.hstack((n_min,a,n_min))
        key, arr = np.unique(a).tolist(), list()
        for n in range(0,b.shape[1]-c-1):
            k = b[:,n:n+(c+2)]
            m = [((k[:,0]!=n) & (k[:,-1]!=n) & 
                  ((k[:,1:-1]==n).sum(axis=1)==c)) for n in key]
            m = tuple([n.astype(int).reshape(-1,1) for n in m])
            arr.append(np.hstack(m))
        return sum(arr), key
    else: return np.full((a.shape[0],len(key)),0), key

def dpd_label(a, b=[30,60,90,float('Inf')]):
    
    '''
    Parameters
    ----------

    a : array-like of shape (n_samples, n_columns)
    \t 2D-array must contain only integers or floats only
    
    b : list of floats or integers, optional, default: [30,60,90,Inf]
    \t Interval of dpds. Range of interval is defined as b[n]<X<=b[n+1]
    '''
    return [((a>n)&(a<=b[m+1]))*n for m,n in enumerate(b[:-1])]

def last_dpd_cnt(a, period=[3,6,9], key=None, col_fmt='last_%dm_%ddpd_cnt'):

    '''
    Parameters
    ----------
    
    a : array-like of shape (n_sample, n_column)
    \t 2D-array must contain only integers or floats only
    
    period : list of integers, optional, default: [3,6,9]
    \t List of observation windows that counting will be taken place. 
    \t Counting starts from right to left and varies according to predefined
    \t observation windows.
    
    key : list, optional, default: None
    \t List of key labels. If None, unique labels are determined from the list
    
    col_fmt : str, optional, default: 'last_%dm_%ddpd_cnt'
    \t The column format that will automatically be generated. 
    \t The first and second positions of '%d' are period and label, respectively 
    
    Return
    ------
    
    dataframe of shape (n_sample, n_key * n_period)
    '''
    columns, arr = list(), list()
    if key is None: key = np.unique(a)
    for p in period: # time period
        for d in key: # labels
            columns.append(col_fmt % (p,d))
            arr.append((a[:,-p:]==d).sum(axis=1).reshape(-1,1))
    return pd.DataFrame(np.hstack((arr)),columns=columns)

def last_consec_dpd_cnt(a, consec=[1,2,3,4], period=[3,6,9], key=None, col_fmt='last_%dm_%ddpd_cs%d_cnt'):
      
    '''
    Parameters
    ----------
    
    a : array-like of shape (n_sample, n_column)
    \t 2D-array must contain only integers or floats only
    
    consec : list of integers, optional, default: [1,2,3,4]
    \t List that contains number of labels appears consecutively

    period : list of integers, optional, default: [3,6,9]
    \t List of observation windows that counting will be taken place. 
    \t Counting starts from right to left and varies according to predefined
    \t observation windows.
    
    key : list, optional, default: None
    \t List of key labels. If None, unique labels are determined from the list
    
    col_fmt : str, optional, default: 'last_%dm_%ddpd_cs%d_cnt'
    \t The column format that will automatically be generated. 
    \t The first, second, and third positions of '%d' are period, label, 
    \t and consecutiveness, respectively 
    
    Return
    ------
    
    dataframe of shape (n_sample, n_key * n_period * n_consecutive)
    '''
    columns, arr = list(), list()
    if key is None: key = np.unique(a)
    for n in consec: # number of consecutiveness
        for p in period: # 
            k, labels = consec_array(a[:,-p:],n,key)
            cols = [col_fmt % (p,d,n) for d in key]
            columns.extend(cols); arr.append(k)
    return pd.DataFrame(np.hstack((arr)),columns=columns)

def diff_dpd_cnt(a, period=[3,6,9], key=None, col_fmt='last_%dm_%dto%d_cnt'):
    
    '''
    Parameters
    ----------
    
    a : array-like of shape (n_sample, n_column)
    \t 2D-array must contain only integers or floats only
    
    period : list of integers, optional, default: [3,6,9]
    \t List of observation windows that counting (delta) will be taken place. 
    \t Counting starts from right to left and varies according to predefined
    \t observation windows. It is noteworthy that period that is more than
    \t width of given array or equals to one, is ignored.
    
    key : list, optional, default: None
    \t List of key labels. If None, unique labels are determined from the list
    
    col_fmt : str, optional, default: 'last_%dm_%dto%d_cnt'
    \t The column format that will automatically be generated. 
    \t The first, second, and third positions of '%d' are period, beginning label, 
    \t and finishing label, respectively 
    
    Return
    ------
    
    dataframe of shape (n_sample, n_key * n_period * n_consecutive)
    '''
    arr, columns = list(), list()
    if key is None: 
        key = np.unique(a).tolist()
        key = [(m,n) for m in key for n in key]
    # remove period that exceeds width of array
    period = [n for n in period if (n <= a.shape[1]) & (n>1)]
    if (a.shape[1]>1):
        for p in period:
            for c in key:
                index = (a[:,-p:]==c[0])+(a[:,-p:]==c[1])
                b = np.where(index==True,a[:,-p:].astype(float),np.nan)
                b = (np.diff(b)==float(np.diff(c))).astype(int).sum(axis=1)
                arr.append(b.reshape(-1,1))
                columns.append(col_fmt % (p,c[0],c[1]))
        return pd.DataFrame(np.hstack((arr)),columns=columns)
    else: return pd.DataFrame(np.zeros((a.shape[0],1)),columns=['error_diff'])
