import pandas as pd, numpy as np, math, time
import matplotlib.pylab as plt
from scipy.stats import spearmanr, pearsonr, sem, t, chi2
import scipy.stats as st, ipywidgets as widgets
from IPython.display import display

#@markdown **_class_** : woe_binning

class woe_binning:
  
  '''
  Method
  ------

  \t (1) self.fit(y, X)
  \t - Fit the model according to the given initial inputs.
  \t **Return** 
  \t - array of bin edges
  \t (2) self._woe_btw_bins(y, X, r_min, r_max, plot=False, X_name=None)
  \t - find series of cut-offs according to the given initial inputs and range  
  \t **Return** 
  \t - self.woe_df (dataframe)
  '''
  def __init__(self, trend=-1, n_order=1, n_step=20, min_pct=0.05, method='iv', chi_alpha=0.05, 
               chi_intv=15, p_value=0.05, ttest_intv=15, min_obs=0.05, min_event=0.05):
    '''
    Parameters
    ----------

    \t trend : (int), predefined trend of WOEs { 0 : downward trend, 1: upward trend, -1: allow function to determine trend (_default_)}
    \t n_order : (int), an order of selection
    \t n_step : (int), number of steps (percentile) given defined range (min, max)
    \t min_pct : (float) minimum percentage of sample in each BIN
    \t mehtod : (str) method of optimization
    \t - 'iv' : determine the cut-off with highest value of information value
    \t      $S_{0} = \sum_{y=0\Subset Y} 1$ and $S_{1} = S - S_{0}$
    \t      $P(R,y) = (\sum_{y\Subset Y|R} 1)/S_{y}$ 
    \t      where $R$ is range of $x$ given cut point, and $y \Subset$ {0, 1} or {non-event, event}
    \t      $IV(T,S) = \sum_{i=1}^{2} (P(R,0))-P(R,1))\log(P(R,0)/P(R,1))$
    \t      where $i$ = nth interval {1, 2}, and  $T$ = cutoff
    \t - 'entropy' : use entropy to determine cut-off that returns the highest infomation gain
    \t      $Ent(S_{i}) = -\sum_{j=1}^{k} P(C_{j},S_{i})\log_{2}(P(C_{j},S_{i}))$
    \t      $E(T,S) = (S_{1}/S) Ent(S_{1}) +  (S_{2}/S) Ent(S_{2}) $
    \t      where $k$ = number of classes in $S_{i}$, $i \Subset$ {1, 2}, $P(C_{j},S_{i})$ = probability of $C_{j}$ given $S_{i}$, 
    \t      $T$ = cutoff, and $S = S_{1}+ S_{2}$
    \t - 'gini' : use gini-impurity to determine cut-off that has the least contaminated groups
    \t      $Gini(S_{i}) = 1-\sum_{j=1}^{k} P(C_{j},S_{i})^{2}$
    \t      $Gini(T,S) = (S_{1}/S) Gini(S_{1}) +  (S_{2}/S) Gini(S_{2}) $
    \t      where $k$ = number of classes in $S_{i}$, $i \Subset$ {1, 2}, $P(C_{j},S_{i})$ = probability of $C_{j}$ given $S_{i}$, 
    \t      $T$ = cutoff, and $S = S_{1}+ S_{2}$
    \t - 'chi' : chi-merge (supervised bottom-up) Using Chi-sqaure, it tests the null hypothesis that two adjacent intervals are independent.
    \t      If the hypothesis is confirmed the intervals are merged into a single interval, if not, they remain separated.
    \t      $\chi^{2} = \sum_{i=1}^{m}\sum_{j=1}^{k} (A_{ij}-E_{ij})^{2}/E_{ij}$
    \t      where $m$ = 2 (2 intervals being compared), $k$ = number of classes, 
    \t      $A$ = actual number of samples, and $E$ = expected number of samples (independent)
    \t - 'mono' : monotonic-optimal-binning. Using Student's t-test (two independent samples), it tests the null hypothesis that means of two 
    \t      adjacent intervals are the same. If the hypothesis is confirmed the intervals remain separated, if not, they are merged into a single interval.

    \t chi_alpha : (float), significant level of Chi-sqaure
    \t chi_intv : (int), starting sub-intervals of Chi-merge
    \t p_value : (float), significant level of Student's T-Test
    \t ttest_intv : (int), starting sub-intervals of "Monotonic-Optimal-Binning" technique
    \t min_obs : (float), minimum percentage of sample in each BIN (used in 'mono')
    \t min_event : (float), minimum percentage of event compared to its total in each BIN
    '''
    # Multi-Interval Discretization (modified)
    self.trend, self.n_order = trend, n_order
    self.n_step, self.min_pct = n_step, min_pct
    self.method = method
    
    # Chi-Merge
    self.chi_alpha, self.chi_intv  = chi_alpha, chi_intv
    
    # Monotone Optimal Binning
    self.p_value, self.ttest_intv = p_value/2, ttest_intv
    self.min_obs, self.min_event = min_obs, min_event

  def fit(self, y, X):
    
    '''
    (1) Multi-Interval Discretization (modified) (method='iv','gini','entropy')
    (2) Chi-Merge (method='chi')
    (3) Monotone Optimal Binning (method='mono')
    '''
    switcher = {'iv': 1, 'gini': 2, 'entropy': 3, 'chi': 4, 'mono': 5}
    n = switcher.get(self.method, 999) 
    if n <= 3: self.__multi_inv_discretize(y, X)
    elif n == 4 : self.__chi_merge(y, X)
    elif n == 5: self.__monotone_optiomal_bin(y, X)
    
  def __monotone_optiomal_bin(self, y, X):    
    
    '''
    Monotone Optimal Binning
    Author: Pavel Mironchyk and Viktor Tchistiakov (rabobank.com)
    (1) For every pair of adjecent bin p-value values is computed
    (2) if number of defaults or number of observation is less than defined limits
    add 1 to p-value. if a bin contains just one observation then set p-value equals to 2.
    (3) Merge pair with highest p-value into single bin
    (4) Repeat (1), (2) and (3) until all p-values are less than critical p-value
    '''
    # Initialize the overall trend and cut point
    nonan_X = X[np.logical_not(np.isnan(X))]
    nonan_y = y[np.logical_not(np.isnan(X))]
    n_target = nonan_y[nonan_y==1].size
    df = self.__n2_df(y, X)
    bin_edges = self.__pct_bin_edges(X, self.ttest_intv) # <-- Intervals
    n_bins = len(bin_edges) + 1
    
    while (len(bin_edges) < n_bins) & (len(bin_edges) > 3):
      n_bins = len(bin_edges)
      p_values = np.full(n_bins-2,0.0)
      for n in range(n_bins-2):
        r_min, cutoff, r_max = bin_edges[n], bin_edges[n+1], bin_edges[n+2]
        interval = (nonan_X>=r_min) & (nonan_X<r_max)
        # number of observations when merged
        n_obs = nonan_X[interval].size
        n_event = nonan_y[interval & (nonan_y==1)].size/n_target
        if n_obs == 1: p_values[n] = 2
        elif (n_obs < self.min_obs) | (n_event < self.min_event): p_values[n] = 1
        else:
          x1 = nonan_X[(nonan_X>=r_min) & (nonan_X<cutoff)]
          x2 = nonan_X[(nonan_X>=cutoff) & (nonan_X<r_max)]
          _, p = self.__independent_ttest(x1, x2) # <-- t-test
          p_values[n] = p
      if max(p_values) > self.p_value:
        p_values = [-float('Inf')] + p_values.tolist() + [-float('Inf')]
        bin_edges = [a for a, b in zip(bin_edges,p_values) if b < max(p_values)]
        
    self.bin_edges = bin_edges   
       
  def __chi_merge(self, y, X):
    
    '''
    Chi-merge
    (1) For every pair of adjecent bin a X2 values is computed
    (2) Merge pair with lowest X2 into single bin
    (3) Repeat (1) and (2) until all X2s are more than predefined threshold
    '''
    # Initialize the overall trend and cut point
    dof = len(np.unique(y)) -1
    threshold = chi2.isf(self.chi_alpha, df=dof)
    p_value = 1-chi2.cdf(threshold, df=dof) # <-- Rejection area
    bin_edges = self.__pct_bin_edges(X, self.chi_intv) # <-- Intervals
    n_bins = len(bin_edges) + 1
    
    while (len(bin_edges) < n_bins) & (len(bin_edges) > 3):
      n_bins = len(bin_edges)
      crit_val = np.full(n_bins-2,0.0)
      for n in range(n_bins-2):
        r_min, cutoff, r_max = bin_edges[n], bin_edges[n+1], bin_edges[n+2]
        c, _ = self.__chi_square(y, X, r_min, r_max, cutoff)
        crit_val[n] = c
      if min(crit_val) < threshold:
        crit_val = [float('Inf')] + crit_val.tolist() + [float('Inf')]
        bin_edges = [a for a, b in zip(bin_edges,crit_val) if b > min(crit_val)]
    self.bin_edges = bin_edges    
 
  def __multi_inv_discretize(self, y, X):
    
    '''
    Multi-Interval Discretization (modified)
    List of cutoffs is determined in this instance basing on 3 different 
    indicators, which are Information Value, Entropy, and Gini Impurity 
    (1) For each interval, divide input into several cut points 
    (2) for each cut point, compute the indicator that satisfies trend
    of WOEs and optimizes objective function. if not, no cutoff is selected
    for this bin.
    (3) Repeat (1) and (2) until no cutoff is made
    '''
    # Initialize the overall trend and cut point
    r_min , r_max = np.nanmin(X), np.nanmax(X) + 1 
    cutoff, self.trend = self.__find_cutoff(y, X , r_min, r_max)
    bin_edges, n_bins = np.unique([r_min, cutoff, r_max]).tolist(), 0

    if len(bin_edges) > 2:
      while len(bin_edges) > n_bins:
        n_bins, new_bin_edges = len(bin_edges), list()
        for n in range(n_bins-1):
          r_min, r_max = bin_edges[n], bin_edges[n+1] 
          cutoff, _ = self.__find_cutoff(y, X , r_min , r_max)
          if cutoff != r_min: new_bin_edges.append(cutoff)
        bin_edges.extend(new_bin_edges)
        bin_edges = np.unique(np.sort(bin_edges,axis=None)).tolist()
    else: bin_edges = np.arange(3)
    self.bin_edges = bin_edges

  def __find_cutoff(self, y, X, r_min, r_max):
    
    '''
    Finds the optimum cut point that satisfies the objective function.
    (optimize defined indicator e.g. IV)
    '''  
    # default values: {cutoff: r_min, corrrelation: -1}
    cutoff, woe_corr = r_min, -1

    # determine WOEs between bins ==> woe_df
    woe_df = self._woe_btw_bins(y, X, r_min, r_max)
    
    # if self._woe_btw_bins returns empty
    if woe_df.shape[0]>0:

      # if trend is not predefined {0: downward, 1: upward}
      if self.trend==-1: 
        woe_corr = woe_df.loc[(woe_df['corr_index']==1),'corr_index'].count()
        woe_corr = woe_corr/woe_df.shape[0]
      else: woe_corr = self.trend

      # select entry that corresponds to woe_corr
      if woe_corr >= 0.5: cond = (woe_df['corr_index']==1)
      else: cond = (woe_df['corr_index']==0)
      b = woe_df.loc[cond,['value','cutoffs']]
      b = b.sort_values(['value'], ascending=False).values[:,1]

      # select cutoff
      if len(b) != 0: cutoff = b[min(self.n_order-1,len(b)-1)]

    return cutoff, woe_corr
   
  def _woe_btw_bins(self, y, X, r_min, r_max, plot=False, X_name=None, 
                    bin_decimal=2, fname=None):
    
    '''
    Determines list of WOEs from different cut points. This is applicable
    only to Multi-Interval Discretization i.e. 'iv', 'entropy' and 'gini'.
    NOTE: disabled functions i.e. plot=False, X_name=None

    Parameters
    ----------

    \t y : array_like shape (n_samples)
    \t X : array-like or list shape (n_samples)
    \t r_min, r_max : (float), minimum and maximum values of the range
    \t plot : (boolean), whether to plot the result or not (default=False)
    \t X_name : (str), name of variable. This appears in the title of the plot
    \t bin_decimal : (int), decimal places displayed in BIN
    \t fname : \str or PathLike or file-like object

    Return
    ------

    \t self.woe_df (dataframe)
    '''
    method = {'iv': 1, 'gini': 1, 'entropy': 1, 'chi': 0, 'mono': 0}
    n = method.get(self.method, 999) 
    if n==1:

      # total number of samples for respective classes
      df = self.__n2_df(y, X); self.X_name = X_name
      n_event = [df.loc[df.target==n,'target'].count() for n in range(2)]
      
      # accept values that are not null and stay within defined boundary
      df = df.loc[(df.variable.notnull()) & (r_min<=df.variable) & (df.variable<r_max)]

      # determine list of cutoffs given number of bins
      bin_cutoff = self.__pct_bin_edges(df['variable'].values, self.n_step)
      left_woe, right_woe, new_bin, v_list = list(), list(), list(), list()
      
      if len(bin_cutoff)-2 > 0:
        for cutoff in bin_cutoff[1:-1]:
          # distributions of event and non-event
          left, right, a, b = self.__2_interval_dist(df, cutoff, n_event)

          # (1) Bin should contain at least 5% observations
          # (2) Bin should not have 0 accounts for good or bad
          if (self.min_pct <= min(a,b)) & (0 < min(left + right)):
            
            # Weight of Evidence
            left_woe.append(self.__woe(left[0], left[1]))
            right_woe.append(self.__woe(right[0], right[1]))
            new_bin.append(cutoff)

            if self.method == 'iv':
              a = self.__iv(left[0], left[1])
              b = self.__iv(right[0], right[1])
              v_list.append(a+b)
              
            elif self.method == 'entropy':
              X, y = df.variable.values, df.target.values
              _ , a = self.__entropy(y, X) # <-- Initial Entropy
              _ , b = self.__entropy(y, X, cutoff=cutoff) # <-- Bin Entropy
              v_list.append(a-b)
        
            elif self.method == 'gini':
              X, y = df.variable.values, df.target.values
              _ , a = self.__gini(y, X) # <-- Initial Gini
              _ , b = self.__gini(y, X, cutoff=cutoff) # <-- Bin Gini
              v_list.append(a-b)
      
      # create result dataframe
      self.woe_df = self.__woe_df(left_woe, right_woe, v_list, new_bin)
      if (plot==True) & (self.woe_df.shape[0]>=1): 
        self.__plot_woe(r_min=r_min, r_max=r_max, 
                        bin_decimal=bin_decimal, fname=fname)
      return self.woe_df
    else:
      print('The method (%s) is not applicable to this instance' % self.method)
      return None

  def __plot_woe(self, r_min, r_max, bin_decimal=2, fname=None):
    
    '''
    Plot Weight-of-Evidence (WOE) from 2 intervals 
    for each respective cut points
    '''
    # Create tick and its labels
    cutoffs = self.woe_df['cutoffs'].values
    x_ticks = np.arange(self.woe_df.shape[0])
    x_ticklbs = np.full(len(x_ticks),0.0)
    for n, cutoff in enumerate(cutoffs):
      if n > 1000: x_ticklbs[n] = '{:.1e}'.format(cutoff)
      else: x_ticklbs[n] = str('{:.' + str(bin_decimal) + 'f}').format(cutoff)
    
    # plot
    fig, axis = plt.subplots(1,1,figsize=(max(len(cutoffs)*0.55,6),4))

    # upward and downward trends (bar chart)
    up_y, dn_y, up_max, dn_max, up_index, dn_index = self.__trend()

    # vertical span
    if len(up_index) > 0:
      axis.axvspan(up_index[0]-0.5, up_index[0]+0.5, color='#32ff7e', alpha=0.3)
      axis.text(up_index[0], 0, 'max', va='top', ha='center')
    if len(dn_index) > 0:      
      axis.axvspan(dn_index[0]-0.5, dn_index[0]+0.5, color='#ff4d4d', alpha=0.3)
      axis.text(dn_index[0], 0, 'max', va='bottom', ha='center')

    # plot WOEs
    kwargs = dict(alpha=0.8, width=0.7, align='center', hatch='////', 
                  edgecolor='#4b4b4b', lw=1)
    bar_1 = axis.bar(x_ticks, up_y, color='#32ff7e', label='up trend', **kwargs)
    bar_2 = axis.bar(x_ticks, dn_y, color='#ff4d4d', label='down trend', **kwargs)
    
    # infomation value or gain
    tw_axis = axis.twinx()
    kwargs = dict(color='#2d3436', lw=1, ls='--', label='Information Value', 
                  marker='o',markersize=4, fillstyle='none')
    lns_1 = tw_axis.plot(x_ticks, self.woe_df['value'].values, **kwargs)
    if self.method=='entropy': label = 'Information Gain (entropy)'
    elif self.method=='gini': label = 'Information Gain (gini impurity)'
    else: label = 'Information Value (IV)'
    tw_axis.set_ylabel(label)
    tw_axis.grid(False)

    # plots and labels
    plots = [bar_1, bar_2, lns_1[0]]
    labels = [bar_1.get_label(), bar_2.get_label(),lns_1[0].get_label()]

    axis.set_facecolor('white')
    axis.set_ylabel(r'$Trend = WOE_{right}-WOE_{left}$')
    axis.set_xlabel('Cut-off')
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_ticklbs)
    axis.set_xlim(-0.5, len(x_ticks)-0.5)
    title = 'WOE comparison - Variable: %s \n ( min=%.2f, max=%.2f, Bin=%d )' 
    axis.set_title(title % (self.X_name, r_min, r_max, len(x_ticks)))
    ylim = axis.get_ylim()
    yinv = float(abs(np.diff(axis.get_yticks()))[0])
    axis.set_ylim(ylim[0]-0.5, ylim[1]+0.5)
    axis.grid(False)
    kwargs = dict(loc='best', fontsize=10, framealpha=0, edgecolor='none')
    axis.legend(plots, labels, **kwargs)

    # change label on secondary y-axis
    y_ticklbs = list()
    for n, value in enumerate(tw_axis.get_yticks()):
      y_ticklbs.append('{:.2e}'.format(value))
    tw_axis.set_yticklabels(y_ticklbs)

    if fname is not None: plt.savefig(fname)
    plt.tight_layout()
    plt.show()
  
  def __trend(self):

    a = self.woe_df.copy()
    a['diff'] = a['right_woe'] - a['left_woe']
    a.loc[a['diff']>=0,'upward'] = a['diff']
    a.loc[a['diff']<0,'downward'] = a['diff']
    up_y, dn_y = a['upward'].values, a['downward'].values
    up_max = a.loc[(a['upward']>=0),'value'].max()
    dn_max = a.loc[(a['downward']<0),'value'].max()
    up_index = a.index[a['value']==up_max].tolist()
    dn_index = a.index[a['value']==dn_max].tolist()
    return up_y, dn_y, up_max, dn_max, up_index, dn_index

  def __2_interval_dist(self, a, cutoff, n_event):
    
    ''' 
    Determines distribution of events and non-events from 2 intervals 
    given cut point, where 0 and 1 represent former and latter, respectively
    NOTE: n_event is number of samples in each class from entire sample
    '''
    X, y = a['variable'], a['target']
    left, right = [None,None], [None,None]
    for n in [0,1]:
      # min <= X_left < cutoff
      cnt = X.loc[(X<cutoff) & (y==n)].count()
      left[n] = cnt/float(n_event[n])
      # cutoff <= X_right < max
      cnt = X.loc[(cutoff<=X) & (y==n)].count()
      right[n] = cnt/float(n_event[n])
    nL = X.loc[(X<cutoff)].count()/sum(n_event) # <-- % left dist.
    nR = X.loc[(cutoff<=X)].count()/sum(n_event) # <-- % right dist.
    return left, right, nL, nR
  
  def __woe_df(self, left_woe, right_woe, value, new_bin):
    
    '''
    Constructs result dataframe from self._woe_btw_bins
    '''
    # create result dataframe
    a = [left_woe, right_woe, value, new_bin]
    a = [np.array(arr).reshape(-1,1) for arr in a]
    a = np.concatenate(a, axis=1)
    a = pd.DataFrame(a,columns=['left_woe','right_woe','value','cutoffs'])
    # woe_corr : {1: positive, 0: negative}
    a['corr_index'], a.loc[(a.left_woe<=a.right_woe),'corr_index'] = 0, 1
    return a
         
  def __pct_bin_edges(self, X, bins=20):
  
    ''' 
    Creates percentile Bins
    NOTE: Each bin should contain at least 5% percent observations
    '''
    nonan_X = X[np.logical_not(np.isnan(X))]
    bin_incr = 100/float(bins)
    bin_edges = [np.percentile(nonan_X, min(n*bin_incr,100)) for n in range(bins+1)]
    bin_edges = np.unique(bin_edges)
    bin_edges[-1:] = bin_edges[-1:] + 1
    return np.unique(bin_edges)
  
  def __entropy(self, y, X, cutoff=None):
  
    '''
    Multi-interval Discretization (entropy)
    Parameters
    (1) y: array or list of classes
    (2) X: array or list of values
    (3) cutoff: cut-off value (float or integer)
        i.e. left-side <= cut-off, right-side > cut-off
        if cutoff equals to None, max value is assigned
    '''
    if cutoff==None: cutoff=max(X)
    n_class, n_cnt = np.unique(y), len(y)
    df = self.__n2_df(y,X)
    
    cond, bin_ent = [(df.variable<=cutoff),(df.variable>cutoff)], np.zeros(2) 
    for n in range(2):
      n_cutoff = max(df.loc[cond[n],'variable'].count(),1)    
      p_cutoff, entropy = n_cutoff/float(n_cnt), 0
      for target in n_class:
        cls_cond = cond[n] & (df.target==target)
        cls_p  = df.loc[cls_cond,'variable'].count()/float(n_cutoff)
        if cls_p>0: entropy += -cls_p*math.log(cls_p,2)
      bin_ent[n] = p_cutoff * entropy
    return bin_ent, sum(bin_ent)
  
  def __gini(self, y , X, cutoff=None):
  
    '''
    Multi-interval Discretization (gini-impurity)
    Parameters
    (1) y: array or list of classes
    (2) X: array or list of values
    (3) cutoff: cut-off value (float or integer)
        i.e. left-side <= cut-off, right-side > cut-off
        if cutoff equals to None, max value is assigned
    '''
    if cutoff==None: cutoff=max(X)
    n_class, n_cnt = np.unique(y), len(y)
    df = self.__n2_df(y, X)
    
    cond = [(df.variable<=cutoff),(df.variable>cutoff)]
    gini_np = np.zeros(2)
    for n in range(2):
      n_cutoff = max(df.loc[cond[n],'variable'].count(),1)    
      p_cutoff, gini = n_cutoff/float(n_cnt), 1
      for target in n_class:
        cls_cond = cond[n] & (df.target==target)
        cls_p = df.loc[cls_cond,'variable'].count()/float(n_cutoff)
        gini += -cls_p**2
      gini_np[n] = p_cutoff * gini
    return gini_np, sum(gini_np)
  
  def __chi_square(self, y, X, r_min, r_max, cutoff):
    
    '''
    Using Chi-Square to test similarity or Goodness-of-Fit
    Null Hypothesis: two intervals are dependent (or similar)
    '''
    df = self.__n2_df(y, X)
    df = df.loc[(df['variable']>=r_min) & (df['variable']<r_max)]
    n_class, n_cnt = np.unique(df['target'].values), len(df)
    dof = len(n_class) - 1
    
    conds, crit_val = [(df.variable<=cutoff),(df.variable>cutoff)], 0
    n_R = [int(df.loc[a,'variable'].count()) for a in conds]
    n_C = [int(df.loc[(df.target==a),'variable'].count()) for a in n_class]
    for n,a in enumerate(conds):
      for m,b in enumerate(n_class):
        C_ab = df.loc[a & (df.target==b),'variable'].count()
        E_ab = n_R[n]*n_C[m]/float(n_cnt)
        if E_ab > 0: crit_val += (C_ab-E_ab)**2/float(E_ab)
    p_value = 1-chi2.cdf(crit_val,df=dof)
    return crit_val, p_value
  
  def __independent_ttest(self, x1, x2):
    
    '''
    Two-sample t-test using p-value 
    Null Hypothesis: mean of two intervals are the same  
    '''
    # calculate means
    mean1, mean2 = np.mean(x1), np.mean(x2)
    
    # calculate standard deviations
    std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    
    # calculate standard errors
    n1, n2 = len(x1), len(x2)
    se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
    sed = np.sqrt(se1**2 + se2**2)
   
    # t-statistic
    #(when sed=0 that means x1 and x2 are constant)
    if sed>0: t_stat = (mean1-mean2) / sed
    else: t_stat = float('Inf')
    
    # calculate degree of freedom
    a, b = se1**2/n1, se2**2/n2
    c, d = 1/(n1-1), 1/(n2-1)
    if (a+b>0): df = math.floor((a+b)/(a*c+b*d))
    else: df = n1 + n2 -2
    
    # one-tailed p-value
    p = 1.0-t.cdf(abs(t_stat), df)
    
    return t_stat, p
  
  def __n2_df(self, y, X):
    
    '''
    Construct a dataframe of target and variable from y and X, respectively
    '''
    y, X = np.array(y).reshape(-1,1), np.array(X).reshape(-1,1) 
    a = np.concatenate((y,X), axis=1)
    return pd.DataFrame(data=(a),columns=['target','variable'])
  
  def __woe(self, a, b):
    
    '''
    {a : dist. of non-events , b : dist. of events}
    '''
    return np.log(a/b)
  
  def __iv(self, a, b):
    
    ''' 
    information value component 
    {a : dist. of non-events , b : dist. of events}
    '''
    return (a-b)*np.log(a/b)
