import numpy as np, matplotlib.pyplot as plt

def odd_scaling(y_proba, pdo=50, odds=1.0, point=200, decimal=0):
    
    '''
    ** Scaling Calculation **
    
    In general, the relationship between odds and scores can be 
    presented as a linear transformation
    
    score = offset + factor*ln(odds) --- Eq. (1)
    score + pdo = offset + factor*ln(2*Odds) --- Eq. (2)
    
    Solving (1) and (2), we obtain
    
    factor = pdo/ln(2) --- Eq. (3)
    offset = score - (factor*ln(odds)) --- Eq. (4)
    
    Parameters
    ----------
    
    y_proba : 1D-array, shape of (n_sample,) 
    \t array of probabilities (0 < proba < 1)
    
    pdo : int, optional, default: 50
    \t pdo stands for "points to double the odds"
    
    odds : float, optional, default: 1.0 (50:50)
    \t this serves as a reference odd where "point" is assigned to
    
    point : int, optional, default: 200
    \t point that assigned to a reference odd
    
    Return
    ------
    
    an array of scores
    '''
    p = np.array(y_proba).reshape(-1,1)
    log_odds = np.log(p/(1-p))
    factor = pdo/np.log(pdo)
    offset = point - (factor*np.log(odds))
    return np.round_((offset + factor*log_odds).reshape(1,-1)[0],decimal)
