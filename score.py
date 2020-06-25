import numpy as np, matplotlib.pyplot as plt

'''
Instance
(1) ood_scaling
(2) score_dist
'''

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
    factor = pdo/np.log(2)
    offset = point - (factor*np.log(odds))
    return np.round_((offset + factor*log_odds).reshape(1,-1)[0],decimal)

def score_dist(y_label, score, bins=10, figsize=(8,4.5), width=0.4,
               label=('Non-Target','Target'), color=('#2ed573','#1e90ff'), rotation=0, fname=None):
    
    '''
    Parameters
    ----------
    
    y_label : 1D-array, shape of (n_sample,)
    \t array of binary classes (0 and 1)
    
    score : 1D-array, shape of (n_sample,)
    \t array of scores. size must match with y_label
    
    bins : int or sequence of scalars, optional, default: 10
    \t If bins is an int, it defines the number of equal-width bins in the given range. 
    \t If bins is a sequence, it defines a monotonically increasing array of bin edges, 
    \t including the rightmost edge, allowing for non-uniform bin widths.
    \t Frequency in each bin is defined as bins[i] <= x < bins[i+1]
    
    figsize : (float,float), optional, default: (8,4.5)
    \t width, height in inches of plot
    
    width : float, optional, default: 0.4
    \t width of bar plot
    
    label : tuple of str, optional, default: ('Non-Target','Target')
    \t labels of binary classes
    
    color : tuple of hex, optional, default: ('#2ed573','#1e90ff')
    \t color of binary classes
        
    rotation : int, optional, default: 0
    \t rotation of x labels
    
    fname : str or PathLike or file-like object
    \t A path, or a Python file-like object (see pyplot.savefig)
    '''
    bins = np.histogram(score,bins=bins)[1]
    hist = [np.histogram(score[y_label==n],bins=bins)[0] for n in np.unique(y_label)]
    fig, axis = plt.subplots(1,1,figsize=figsize)
    x = np.arange(len(bins)-1); x_tick = [x-width/2, x+width/2]
    for n in range(2):
        kwargs = dict(label=label[n], width=width, alpha=0.5, color=color[n], edgecolor='#718093')
        axis.bar(x_tick[n], hist[n]/sum(hist[n])*100, **kwargs)
    kwargs = dict(fontsize=12, color='k')
    axis.set_ylabel('Percentage of samples (%)', **kwargs)
    axis.set_xlabel('Score Range', **kwargs)
    axis.set_xticks(x)
    axis.set_xticklabels([r'$\geq${:,.0f}'.format(n) for n in bins[:-1]], fontsize=9, rotation=rotation)
    axis.set_xlim(-0.5,len(x)-0.5)
    kwargs = dict(facecolor=None,framealpha=0,loc='best')
    axis.legend(**kwargs)
    axis.set_title('Distribution of Scores', fontsize=14)
    fig.tight_layout()
    if fname is not None: plt.savefig(fname)
    plt.show()
