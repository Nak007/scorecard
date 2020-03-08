import pandas as pd, numpy as np, math, time, inspect
import matplotlib.pylab as plt
from scipy.stats import spearmanr, pearsonr, sem, t, chi2
import scipy.stats as st, ipywidgets as widgets
from IPython.display import HTML, display

class woe_binning:
  
    '''
    ** Weight of Evidence **
    The WOE framework in credit risk is based on the 
    following relationship:

        log(P(Y=0|X)/P(Y=1|X)) <-- log-odds given X 
    =   log(P(Y=0)/P(Y=1))     <-- Sample log-odds
    +   log(f(X|Y=0)/f(X|Y=1)) <-- Weight of Evidence
    
    where f(X|Y) denotes the conditional probability density 
    function or a discrete probability distribution if X is 
    categorical.
    
    In addition, when WOE is positive the chance of observing Y=1 
    is below average (for the sample), and vice versa when WOE is 
    negative. Basing on provided bins, X should have a monotonic 
    relationship (linear) to the dependent variable (Y).
    
    Methods
    -------
    self.fit(y, X)
    \t Fit the model according to the given initial inputs.
    **Return** 
    \t - array of bin edges
    
    self._woe_btw_bins(y, X, r_min, r_max, plot=False, X_name=None)
    \t find series of cut-offs according to the given initial inputs and range  
    **Return** 
    \t - self.woe_df (dataframe)
    '''
    def __init__(self, trend=-1, n_order=1, n_step=20, min_pct=0.05, method='iv', chi_alpha=0.05, 
                 chi_intv=15, p_value=0.05, ttest_intv=15, min_obs=0.05, min_event=0.05):
        '''
        Parameters
        ----------
        trend : int, optional, default: -1 
        \t Predefined trend of WOEs
        \t  0 : downward trend
        \t  1 : upward trend 
        \t -1 : allow function to determine trend automatically
        
        n_order : int, optional, default: 1 
        \t Order of selection
        
        n_step : int, optional, default: 20 
        \t Number of steps (percentile) given defined range (min, max)
        
        min_pct : float, optional, default: 0.05 (5%)
        \t minimum percentage of samples in each BIN
        
        method : str, optional, default: 'iv'
        \t method of optimization
        \t - 'iv' : determine the cut-off with highest value of information value
        \t - 'entropy' : use entropy to determine cut-off that returns the highest infomation gain
        \t - 'gini' : use gini-impurity to determine cut-off that has the least contaminated groups
        \t - 'chi' : chi-merge (supervised bottom-up) Using Chi-sqaure, it tests the null hypothesis that two adjacent 
        \t   intervals are independent. If the hypothesis is confirmed the intervals are merged into a single interval, 
        \t   if not, they remain separated.
        \t - 'mono' : monotonic-optimal-binning. Using Student's t-test (two independent samples), it tests the null hypothesis 
        \t   that means of two adjacent intervals are the same. If the hypothesis is confirmed the intervals remain separated, 
        \t   if not, they are merged into a single interval.

        chi_alpha : float, optional, default: 0.05 (5%)
        \t significant level of Chi-sqaure (rejection region)
        
        chi_intv : int, optional, default: 15
        \t starting sub-intervals of Chi-merge
        
        p_value : float, optional, default: 0.05 (5%)
        \t significant level of Student's T-Test (rejection region)
        
        ttest_intv : int, optional, default: 15 
        \t starting sub-intervals of "Monotonic-Optimal-Binning" technique
        
        min_obs : float, optional, default: 0.05 (5%)
        \t Minimum percentage of samples (event+non-event) in each BIN 
        \t (used in 'mono')
        
        min_event : float, optional, default: 0.05 (5%)
        \t Minimum percentage of event and non-event samples with respect 
        \t to their totals in each BIN
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
        fit model given pre-determined model
        
        (1) Multi-Interval Discretization (modified) (method='iv','gini','entropy')
        (2) Chi-Merge (method='chi')
        (3) Monotone Optimal Binning (method='mono')
        '''
        # check minimum between self.min_event and 1 
        self.min_event = max(int(self.min_event*(y==1).sum()),1)/sum(y)
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
            add 1 to p-value. if a bin contains just one observation then set p-value 
            equals to 2.
        (3) Merge pair with highest p-value into single bin
        (4) Repeat (1), (2) and (3) until all p-values are less than critical p-value
        '''
        # Initialize the overall trend and cut point
        nonan_X, nonan_y = X[~np.isnan(X)], y[~np.isnan(X)]
        n_target = nonan_y[nonan_y==1].size
        df = self.__n2_df(y, X)
        bin_edges = self.__pct_bin_edges(X, self.ttest_intv) # <-- Intervals
        n_bins = len(bin_edges) + 1

        while (len(bin_edges) < n_bins) & (len(bin_edges) > 3):
            n_bins = len(bin_edges); p_values = np.full(n_bins-2,0.0)
            for n in range(n_bins-2):
                r_min, cutoff, r_max = bin_edges[n], bin_edges[n+1], bin_edges[n+2]
                interval = (nonan_X>=r_min) & (nonan_X<r_max)

                # number of observations from both sides of cutoff
                x1 = nonan_X[(nonan_X>=r_min) & (nonan_X<cutoff)]
                x2 = nonan_X[(nonan_X>=cutoff) & (nonan_X<r_max)]
                
                # number of observations when merged
                n_obs = nonan_X[interval].size
                min_obs = min(n_obs, x1.size, x2.size)
                
                # percent distribution of events
                pct_event = nonan_y[interval & (nonan_y==1)].size/max(n_target,1)

                if min_obs<=1: p_values[n] = 2
                elif (n_obs<self.min_obs)|(pct_event<self.min_event): p_values[n] = 1
                else: p_values[n] = self.__independent_ttest(x1, x2)[1]

            if max(p_values) > self.p_value:
                p_values = [-float('Inf')] + p_values.tolist() + [-float('Inf')]
                bin_edges = [a for a, b in zip(bin_edges,p_values) if b < max(p_values)]

        self.bin_edges = bin_edges   
       
    def __chi_merge(self, y, X):

        '''
        Chi-merge
        (1) For every pair of adjecent bin a X2 values is computed
        (2) Merge pair with lowest X2 (highest p-value) into single bin
        (3) Repeat (1) and (2) until all X2s are more than predefined threshold
        Note: minimum number of bins is 2
        '''
        # Initialize the overall trend and cut point
        dof = len(np.unique(y)) - 1
        threshold = chi2.isf(self.chi_alpha, df=dof)
        p_value = 1-chi2.cdf(threshold, df=dof) # <-- Rejection area
        bin_edges = self.__pct_bin_edges(X, self.chi_intv) # <-- Intervals
        n_bins = len(bin_edges) + 1

        while (len(bin_edges) < n_bins) & (len(bin_edges) > 2):
            n_bins = len(bin_edges)
            crit_val = np.full(n_bins-2,0.0)
            for n in range(n_bins-2):
                r_min, cutoff, r_max = bin_edges[n], bin_edges[n+1], bin_edges[n+2]
                crit_val[n] = self.__chi_square(y, X, r_min, r_max, cutoff)[0]
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
        Note: minimum number of bins is 2
        '''
        # Initialize the overall trend and cut point
        r_min , r_max = np.nanmin(X), np.nanmax(X) * 1.01
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
        else: bin_edges = [r_min, np.median(X[~np.isnan(X)]) ,r_max]
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
   
    def _woe_btw_bins(self, y, X, r_min, r_max, plot=False, X_name=None, bin_decimal=2, fname=None):

        '''
        Determines list of WOEs from different cut points. This is applicable
        only to Multi-Interval Discretization i.e. 'iv', 'entropy' and 'gini'.
        NOTE: disabled functions i.e. plot=False, X_name=None

        Parameters
        ----------

        y, X : array_like of list shape of (n_samples)

        r_min, r_max : float
        \t minimum and maximum values of the range
        
        plot : bool, optional, default: False
        \t whether to plot the result or not
        
        X_name : str, optional, default: None 
        \t name of variable. This appears in the title of the plot
        
        bin_decimal : int, optional, default: 2 
        \t decimal places displayed in BIN
        
        fname : str or PathLike or file-like object, optional, default: None
        \t file path with file name and extension

        Return
        ------

        self.woe_df (dataframe)
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
                    if (self.min_pct <= min(a,b)) & (self.min_event <= min(left + right)):

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
                self.__plot_woe(r_min=r_min, r_max=r_max, bin_decimal=bin_decimal, fname=fname)
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
        x_ticklbs = np.full(len(x_ticks),None)
        for n, cutoff in enumerate(cutoffs):
            if cutoff > 1000: x_ticklbs[n] = str('{:.' + str(bin_decimal) + 'e}').format(cutoff)
            else: x_ticklbs[n] = str('{:,.' + str(bin_decimal) + 'f}').format(cutoff)

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
        kwargs = dict(alpha=0.8, width=0.6, align='center', hatch='////', edgecolor='#4b4b4b', lw=0.8)
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
        axis.set_xticklabels(x_ticklbs, rotation=45)
        axis.set_xlim(-0.5, len(x_ticks)-0.5)
        
        title = tuple((r'$\bf{WOE}$ $\bf{comparison}$ : %s' % self.X_name, 
                       r'( $\bf{min}$=%s, $\bf{max}$=%s, $\bf{Bin}$=%d )' % 
                       ('{:,.2f}'.format(r_min), '{:,.2f}'.format(r_max), len(x_ticks))))
        axis.set_title('\n'.join(title))
        ylim = axis.get_ylim()
        yinv = float(abs(np.diff(axis.get_yticks()))[0])
        axis.set_ylim(ylim[0]-0.5, ylim[1]+0.5)
        axis.grid(False)
        kwargs = dict(loc='best', fontsize=10, framealpha=0, edgecolor='none')
        axis.legend(plots, labels, **kwargs)

        # change label on secondary y-axis
        y_ticklbs = list()
        for n, value in enumerate(tw_axis.get_yticks()):
            y_ticklbs.append('{:.1e}'.format(value))
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
        - Each bin should contain at least 5% of observations
        - In case when X contains the same value more than 90% or more, this
          affects percentile binning to provide an unreasonable range of data.
          Thus, it will switch to equal-binning method to group those with the 
          same value into one bin while the others will spread out.
        '''
        a, b = X[~np.isnan(X)], 100/float(bins)
        bin_edges = [np.percentile(a, min(n*b,100)) for n in range(bins+1)]
        bin_edges = np.unique(bin_edges)
        bin_edges[-1:] = bin_edges[-1:] + 1
        if len(bin_edges) <=2: bin_edges = self.__equal_bin(a,3)
        return bin_edges
  
    def __equal_bin(self, x, method=0):

        '''
        Methods
        -------

        (1) Square-root choice: takes the square root of the number of data 
            points in the sample 
        (2) Sturges' formula is derived from a binomial distribution and implicitly 
            assumes an approximately normal distribution
        (3) The Rice Rule is presented as a simple alternative to Sturges's rule
        (4) Doane's formula is a modification of Sturges' formula, which attempts 
            to improve its performance with non-normal data.
        (5) Scott's normal reference rule
        (6) Freedman–Diaconis' choice
        '''
        a = x[~np.isnan(x)].copy()
        v_max, v_min, v_sum = max(a), min(a), sum(a)
        stdev, median, n = np.std(a), np.median(a), len(a)
        method = max(min(method,6),1)

        if method==1: 
            bins = math.sqrt(n)
        elif method==2: 
            bins = math.ceil(math.log(n,2) + 1)
        elif method==3:                  
            bins = 2*(n**(1/3))
        elif method==4:
            g = abs(((v_sum/n)-median)/stdev)
            n_stdev = (6*(n-2)/((n+1)*(n+3)))**0.5
            bins = 1 + math.log(n,2) + math.log((1+g/n_stdev),2)
        elif method==5:
            bin_width = 3.5*stdev/(n**(1/3))
            if bin_width > 0: bins = (v_max-v_min)/bin_width
            else: bins = math.ceil(math.log(n,2) + 1)
        elif method==6:
            p25, p75 = np.percentile(a,25), np.percentile(a,75)
            bin_width = 2*(p75-p25)/(n**(1/3))
            if bin_width > 0: bins = (v_max-v_min)/bin_width
            else: bins = math.ceil(math.log(n,2) + 1)

        # round up number of bins 
        bins = max(int(math.ceil(bins)),2)
        bin_width = (v_max-v_min)/bins
        bin_edges = [min(v_min+(n*bin_width),v_max) for n in range(bins+1)]
        bin_edges[-1] = bin_edges[-1] + 1
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
        Null hypothesis: two intervals are dependent (or similar)
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
        Null hypothesis: mean of two intervals are the same  
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

class batch_evaluation:

    '''
    Batch process of woe binning
    
    Parameters
    ----------    
    iv_imp : float, optional, default: 0.5 (50%)
    \t Information Value (IV) importance weight (max=1)

    min_iv : float, optional, default: 0.1 
    \t minimum acceptable Infomation Value (IV)

    min_corr : float, optional, default: 0.5
    \t minimum acceptable absolute correlation

    max_tol : float, optional, default: 0.01 (1%) 
    \t maximum tolerance of difference between model and 
    \t log(event/non-event) intercepts
        
    Method
    ------
    self.fit
    \t Fit the model according to inputs
    
    self.rank_methods
    \t Determine best binning method for each variable
    
    Attributes
    ----------
    self.n_missing : list of str
    \t list of variable(s) that contains nothing

    self.constant : list of str
    \t list of variable(s), apart from missing,  
    \t contains only one value (like a constant) 
    '''
    def __init__(self, iv_imp=0.5, min_iv=0.1, min_corr=0.5, max_tol=0.01, **kwargs):
        
        # Keyword arguments
        self.kwargs = self.__match_kwargs(woe_binning,kwargs)
        k = dict([('iv_imp',iv_imp), ('min_iv',min_iv), 
                  ('min_corr',min_corr), ('max_tol',max_tol)])
        self.rank = self.__match_kwargs(batch_evaluation.rank_methods,k)
        
        # Define method
        k = self.kwargs['method']
        if isinstance(k,str): self.method = list([k])
        elif isinstance(k,list): self.method = k
        else: self.method = ['iv','entropy','gini','chi','mono']
            
    def fit(self, X, y):

        '''
        Parameters
        ----------
        y : array-like of shape (n_samples,)
        \t Target values (binary)
        
        X : array-like or sparse matrix of shape (n_samples,n_features)
        \t Training data
        
        Returns
        -------
        self.data : dictionary of dataframe object
        \t keys of self.data are
        \t (1) 'log'  : output measurements of each iteration
        \t (2) 'data' : binning results
        \t (3) 'rank' : best binning method for each variable
        '''
        # Assign value to variables
        self.__widgets() 
        y = np.array(y) 
        X, features = self.__to_df(X)
        t, n, log = len(features)*len(self.method) , 0, []

        # Set label format
        progress = ' Method : %s , Variable : %s'
        columns = [['round', 'method', 'variable', 'model_bin', 'iv', 
                    'correlation', 'intercept'],
                   ['round', 'variable', 'min','max','bin', 'non_events', 
                    'events', 'pct_nonevents', 'pct_events', 'woe', 'iv']]
        data = pd.DataFrame(columns=columns[1])
        self.data = dict([('log',None),('data',None),('rank',None)])
        iterations = [(m,f) for m in self.method for f in features]
        
        for n_iter in iterations:
      
            # update progress bar
            n += 1; self.__update(n/t, progress % n_iter )

            # WOE binning given selected method
            k = {**self.kwargs, **dict([('method',n_iter[0])])}
            woe1 = woe_binning(**k)
            woe1.fit(y, X[n_iter[1]])
            woe2 = evaluate_bins()
            woe2.fit(X[n_iter[1]], y, woe1.bin_edges, 0.5)

            # Iteration logs
            log.append([n, n_iter[0], n_iter[1], len(woe2.woe_df)-1, 
                        woe2.iv, woe2.rho, woe2.intercept_[2]])

            # Binning results
            a = woe2.woe_df.copy()
            a['round'], a['variable'] = n, n_iter[1]
            data = data.append(a[list(data)], ignore_index=True)
                
        log = pd.DataFrame(log,columns=columns[0])
        self.data['log'] = log.to_dict(orient='list')
        self.data['data'] = data.to_dict(orient='list')
        self.data['rank'] = self.rank_methods(**self.rank)

    def rank_methods(self, iv_imp=0.5, min_iv=0.1, min_corr=0.5, max_tol=0.01):
        
        '''
        Before applying importance weight, each iteration
        must pass all following criteria,
        (1) exceeds minimum Information Value (IV) or not
            equal to np.nan (missing)
        (2) exceeds minimum absolute correlation
        (3) remain within maximum tolerance for intercept

        Subsequently, importance weight is applied and its 
        score is expressed as follows
        
            score(n) = (w)*iv(n) + (1-w)*abs(rho(n))
        
        n   : nth round (iteration)
        w   : importance weight of iv (Information Value)
        rho : Pearson correlation coefficient

        Last but not least, if there is more than one output 
        (per variable) left as abovementioned, the earliest 
        'round' is selected.
        '''
        # set out conditions
        data = pd.DataFrame(self.data['log'].copy())
        data = data.loc[~data['iv'].isna()]
        cond = (data['iv']>=min_iv)
        cond = cond & (abs(data['correlation'])>=min_corr)
        cond = cond & (data['intercept']<=max_tol)
        data = data.loc[cond]

        # weighted score between correlation and iv
        data['w_scr'] = + data['iv']*iv_imp + abs(data['correlation'])*(1-iv_imp) 

        # select method, whose weighted score is the highest
        a = data[['variable','w_scr']].groupby(['variable']).agg(['max'])
        a = pd.DataFrame(a.reset_index().values, columns=['variable','max_scr'])
        a = data.merge(a, on='variable', how='left')
        a = a.loc[(a['w_scr']==a['max_scr'])].drop(columns=['w_scr','max_scr'])

        # if there is more than one candidate left, select the first one
        a = a[['variable','round']].groupby(['variable']).agg(['min'])
        a = a.values.reshape(1,-1).tolist()[0]
        data = data.loc[(data['round'].isin(a))].drop(columns=['w_scr'])
        return data.reset_index(drop=True).to_dict(orient='list')

    def plot(self, column='round', value=1, adjusted=False, **kwargs):

        '''
        Plot WOE (Weight of Evidence) from selected list. 
        Column selection can ONLY be made from either 
        'round' or 'varialbe'. Value must correpond to 
        selected columns (str, int, or list-like)

        Parameters
        ----------
        column : str, optional, default:'round'
        \t It can ONLY be either 'round' or 'variable'
        
        value : str or int or list, optional, default: 1 
        \t If value is an int (column='round'), it defines 
        \t the round number, whereas it defines variable
        \t name, when value is str (column='variable'),
        \t value can be a list of either int or st
    
        adjusted : bool, optional, default: False 
        \t if True, self.data['rank'] is used instead.
        
        **kwargs : keyword arguments
        \t Initial keyword arguments for "plot_woe" class,
        \r and "plot_woe.show()" method. In additon, all
        \t items do not have to be arranged in any 
        \t particular order. Matching will be carried out
        \t automatically.
        '''
        if isinstance(value,(str, int)): value = np.array([value])
        if column not in ['round','variable']: column = 'round' 

        # select list   
        if adjusted: log = pd.DataFrame(self.data['rank'])
        else: log = pd.DataFrame(self.data['log'])
        data = pd.DataFrame(self.data['data'])
        k1 = self.__match_kwargs(plot_woe,kwargs)
        k2 = self.__match_kwargs(plot_woe.show,kwargs)
        p, c = plot_woe(**k1), ['method','variable','correlation']

        for n in value:
            a = log.loc[(log[column]==n)]
            n_round = np.unique(a['round'])
            if n_round.size > 0:
                for m in n_round:
                    b = a.loc[(a['round']==m),c].values[0]
                    df = data.loc[(data['round']==m)]
                    print('Method: %s , Round: %d' % (b[0], m))
                    k2['var_name'] = b[1]; k2['rho'] = b[2]
                    p.show(df,**k2)
            else: print('[Empty list]: %s' % n)
  
    def __widgets(self):

        kwargs = dict(value=0, min=0, max=100, step=0.01, bar_style='info', 
                      orientation='horizontal')
        self.w_t = widgets.HTMLMath(value='Calculating...')
        self.w_f = widgets.FloatProgress(**kwargs)
        w = widgets.VBox([self.w_t, self.w_f])
        display(w); time.sleep(5)
  
    def __update(self, pct, label=None):

        self.w_f.value = pct*100
        self.w_t.value = '({:.0f}%) '.format(pct*100) + label 
        time.sleep(0.1)
        if pct == 1: 
            self.w_f.bar_style = 'success'
            self.w_t.value = '(100%) Complete'
      
    def __to_df(self, X):

        '''
        (1) if X is an array with shape of (n_sample, n_features), 
            it will be transformed to dataframe. Name(s) will be 
            automatically assigned to all columns i.e. X1, X2, etc.
        (2) Remove variable(s) that contains only nan (missing) 
            and will be kept in self.n_missing 
        (3) Remove variable(s), apart from nan (missing), contains 
            only one value (like a constant) and will be kept in 
            self.n_constant
        '''
        if isinstance(X, pd.core.series.Series):
            X = pd.DataFrame(X)
        elif isinstance(X, list):
            X = pd.DataFrame(data=np.array(X),columns='X0')
        elif isinstance(X, np.ndarray):
            columns = ['X'+str(n+1) for n in range(X.shape[1])]
            X = pd.DataFrame(data=np.array(X),columns=columns)
        
        # Remove variable(s) that contains only nan
        features, n_missing, n_constant = list(), list(), list()
        for var in X.columns:
            nan = X[var].isna().sum()/X.shape[0]
            unq = X.loc[~X[var].isna(),var].unique()
            if nan==1: n_missing.append(var)
            elif len(unq)<2: n_constant.append(var)  
            else: features.append(var)
        self.n_missing, self.constant = n_missing, n_constant
        return X[features], features
    
    def __match_kwargs(self, fnc, kw):
        
        '''
        This function only changes keyword arguments that
        match with the function's keyword arguments
        
        Parameters
        ----------
        fnc : function object
        
        kw : dict
        \t Dictionary of keyword arguments
        '''
        f = inspect.getfullargspec(fnc)
        f = dict([(a,b) for (a,b) in zip(f[0][-len(f[3]):],f[3])])
        a = dict([(n,kw[n]) if n in list(kw.keys()) 
                  else (n,f[n]) for n in f.keys()])
        return a
      
class evaluate_bins:
    
    '''
    This function evaluates the predictiveness of bin intervals by 
    using Weight-of-Evidence (WOE), Information Value (IV),
    Spearman rank-order Correlation, and intercept from regression 
    model (compared against log(event/non-event))
    
    Methods
    -------
    self.fit
    \t Fit the model according to inputs
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y, bins=10, replace_null=0.5):
        
        '''
        Parameters
        ----------
        X : array-like or pd.core.series.Series of shape (n_samples,)
        \t Raw data that will be binned according to bin edges

        y : array-like of shape (n_samples,)
        \t Target values (binary)
        
        bins : int or sequence of scalars or str, optional, default:10
        \t If bins is an int, it defines the number of equal-width bins 
        \t in the given range. If bins is a sequence, it defines a 
        \t monotonically increasing array of bin edges, including the 
        \t rightmost edge.
        
        replace_null : float, optional, default:0.5
        \t For any given range, if number of samples for either classes 
        \t equals to "0", it will be replace by replace_null. This is 
        \t meant to avoid error when WOE is calculated

        Returns 
        -------
        self.woe_df : dataframe object
        
        self.iv : float
        \t Information Value
        
        self.rho : float
        \t Spearman rank-order correlation. Exluding missing, it 
        \t measures the correlation between "X" against Weight-of-Evidence.
        \t If WOE ranks monotonically, correlation must be high (>0.5).
        
        self.intercept_ : (float,float,float) 
        \t Tuple of floats, which are arranged as following   
        \t (1) model intercept obtained from fitting y and X
        \t (2) log of event over non-event (intercept)
        \t (3) absolute difference between intercepts
        
        Example
        -------
        >>> model = evaluate_bins()
        >>> model.fit(X, y, bins)
        >>> model.plot() # visualize results
        >>> model.woe_df # WOE dataframe
        >>> model.iv # Information Value
        '''
        if isinstance(X, pd.core.series.Series):
            self.variable = X.name; X = X.values.copy()
        else: self.variable = 'None'
        
        # Weight of Evidence dataframe
        self.woe_df = self.__woe(y, X, bins, replace_null)
        
        # Information Value
        self.iv = self.woe_df['iv'].sum()
        
        # Spearman rank-order 
        self.rho = self.__corr(X)
        
        # Check binning soundness (intercpet)
        self.intercept_ = self.__check_binning(y, X)

    def __woe(self, y, X, bins=10, replace_null=0.5):
        
        # Assign group index to value array, where 0 represents missing
        bins = np.histogram(X[~np.isnan(X)],bins=bins)[1]
        a = np.digitize(X, bins, right=False)
        a[a==len(bins)], cnt = 0, [None]*2
        
        # Determine count, percentage, woe and iv for respective bins
        counts = [np.unique(a[y==c],return_counts=True) for c in [0,1]]
        for (n,a) in enumerate(counts):
            z = np.full(len(bins),0)
            z[np.isin(np.arange(len(bins)),a[0])]=a[1]
            cnt[n] = z.reshape(-1,1).copy()
        pct = [(n/sum(n)) for n in cnt]
        adj_pct =[np.where(n/sum(n)==0,replace_null/sum(n),n/sum(n)) for n in cnt]
        woe = np.where((adj_pct[0]==0) & (adj_pct[1]==0),0,np.log(adj_pct[0]/adj_pct[1]))
        iv = (adj_pct[0]-adj_pct[1])*woe
        
        # BIN Intervals (min,max) 
        n_rng =  [list(l)+[n] for (n,l) in enumerate(zip(bins[:-1],bins[1:]),1)]
        n_rng = np.array([[np.nan]*2+[0]] + n_rng)
        n_rng[1,0], n_rng[-1,1] = -float('Inf'), float('Inf')
        
        # WOE dataframe
        data = np.hstack([n_rng] + cnt + pct + [woe] + [iv])
        columns = ['min','max','bin','non_events','events','pct_nonevents','pct_events','woe','iv']
        return pd.DataFrame(data,columns=columns)

    def __corr(self, X):

        '''
        Spearman rank-order correlation coefficient
        (excluding missing or np.nan)
        
        This function measures the correlation between
        "X" against Weight-of-Evidence
        '''
        woe = self.__assign_woe(X)[~np.isnan(X)].reshape(-1)
        a = X[~np.isnan(X)].reshape(-1)
        try: return spearmanr(a, woe)[0]
        except: return np.nan
    
    def __assign_woe(self, X):

        # determine min and max
        woe_df = self.woe_df.copy()
        nan_woe = float(woe_df.loc[woe_df['bin']==0,'woe'].values)
        bins = (woe_df['min'].values.tolist() + [float('Inf')])[1:]

        # Assign group index to value array and convert to dataframe
        X = np.digitize(X, bins, right=False)
        X = pd.DataFrame(data=X, columns=['bin'])
        X.loc[X['bin']==len(bins),'bin'] = 0
        X = X.merge(woe_df[['bin','woe']], on=['bin'], how='left')
        return X['woe'].values.reshape(-1,1)
    
    def __check_binning(self, y, X):

        '''
        If the slope (Beta) is not 1 or the intercept is not 
        ln(event/non-event), then it could be that binning algorithm 
        is not working properly.
        
        ** Note **
        Due to regularization (l1), this logistic regression will 
        never return coefficient as normal logistic does
        '''
        from sklearn.linear_model import LogisticRegression
        kwargs = dict(solver='liblinear', fit_intercept=True, penalty='l1')
        logit = LogisticRegression(**kwargs)
        
        X = self.__assign_woe(X)
        ln_intercpt = np.log(y.sum()/(len(y)-y.sum()))
            
        try:
            logit.fit(X, y)
            intercept_ = float(logit.intercept_)
        except: intercept_ = np.nan
        diff = abs((intercept_-ln_intercpt)/ln_intercpt)
        return intercept_, ln_intercpt, diff
      
class plot_woe:

    '''
    This functon helps user to visualize arrangement of WOEs 
    (Weight of Evidence) given pre-determined intervals, 
    whereas missing or np.nan value is binned separately.
            
    Parameters
    ----------
    color : list or tuple of color-hex codes, optional, 
    default:['#fff200','#aaa69d','#aaa69d','#eb2f06']
    \t List of color-hex codes, whose items are arranged in the
    \t following order i.e. positive-woe (also non-event), 
    \t negative-woe (also event), sample distribution, and
    \t target rate
        
    label : tuple of str, optional, (default:('non_event','event'))
    \t Tuple of labels used in plots, which are non-target, and
    \t target, respectively.
    
    decimal : int, optional, (default:1)
    \t Number of decimal places for tick labels in x-axis
        
    Method
    ------
    self.show
    \t WOE and distribution plots with respect to pre-determined 
    \t intervals
    '''
  
    def __init__(self, color=['#fff200','#aaa69d','#aaa69d','#eb2f06'], 
                 label=('non_event','event'), decimal=1):

        # Default Keyword Arguments
        self.dm = str(decimal)
        kwargs = dict([('bar',    dict(alpha=0.7, width=0.7, align='center', hatch='////', 
                                       edgecolor='#4b4b4b', lw=1)), 
                       ('text',   dict(va='top', ha='center', color='#4b4b4b', fontsize=10)), 
                       ('line',   dict(marker='o', markersize=5, lw=1, ls='--',fillstyle='none')), 
                       ('legend', dict(loc='best', fontsize=10, framealpha=0, edgecolor='none'))])
        
        # ** WOE plots **
        kwargs['p_woe'] = {**kwargs['bar'], **dict(color=color[0], label='%s>%s'%label)}
        kwargs['n_woe'] = {**kwargs['bar'], **dict(color=color[1], label='%s<%s'%label)}

        # Keyword arguments (bar annotation) 
        kwargs['p_text'] = {**kwargs['text'], **dict(va='top')}
        kwargs['n_text'] = {**kwargs['text'], **dict(va='bottom')}
        
        # ** Distribution plot **
        kwargs['non_event'] = {**kwargs['bar'], **dict(color=color[0], label=label[0])}
        kwargs['event'] = {**kwargs['bar'], **dict(color=color[1], label=label[1])}
        
        # ** Bad Rate plot **
        kwargs['sample'] = {**kwargs['bar'],**dict(color=color[2], label='(%) sample')}
        kwargs['target'] = {**kwargs['line'],**dict(color=color[3], label='(%) {0}'.format(label[1]))}
        self.kw = kwargs
        
    def show(self, woe_df, var_name=None, rho=None, figsize=(5,4.5), fname=None):

        '''
        Parameters
        ----------
        woe_df : dataframe object with mandatory fields
        (1) 'min','max': BIN intervals (min <= x < max)
        (2) 'bin': BIN number
        (3) 'non_events','events':
            number of samples given class in each BIN
        (4) 'pct_nonevents','pct_events': 
            discrete distribution given class in each BIN
        (5) 'woe': Weight of Evidence
        (6) 'iv': Information value
        
        var_name : str, optional, default: None
        \t Variable name. If not defined "None" is applied
        
        rho : float, optional, default: None
        \t Pearson correlation coefficient
        
        figsize: (float, float), optional, default: (5,4.5)
        \t width, height in inches per plot
        
        fname : str or PathLike or file-like object
        \t file path along with file name (*.png)
        
        Returns
        -------
        plots of Weight-of-Evidence, distribution of event and
        non-event, and target-rate
        '''
        self.df = woe_df.rename(str.lower,axis=1).copy()
        if var_name is not None: self.var_name = var_name
        else: self.var_name = 'None'
            
        if rho is None: self.rho = 'None'
        else: self.rho = '%.2f'%rho
        self.iv = self.df['iv'].sum()

        size = (figsize[0]*3, figsize[1])
        fig, axes = plt.subplots(1,3,figsize=size)
        self.__ticklabels()
        self.__woe_plot(axes[0])
        self.__distribution_plot(axes[1])
        self.__target_rate_plot(axes[2])
        
        if fname is not None: plt.savefig(fname)
        plt.tight_layout()
        plt.show()
    
    def __iv_predict(self):

        '''
        Information Value (IV) predictiveness
        It measures the strength of that relationship
        '''
        if self.iv < 0.02: return 'Not useful for prediction'
        elif self.iv >= 0.02 and self.iv < 0.1: return 'Weak predictive Power'
        elif self.iv >= 0.1 and self.iv < 0.3: return 'Medium predictive Power'
        elif self.iv >= 0.3: return 'Strong predictive Power'

    def __ticklabels(self):

        '''
        Set tick labels format (X-axis)
        '''
        ticklabels = np.empty(len(self.df), dtype='|U100')
        f = [''.join((r'$\geq${:.',self.dm,'%s}'%n)) for n in ['f','e']]
        a = np.array([n for n in self.df['min'] if ~np.isnan(n)])
        labels = [f[0].format(x) if abs(x)<1000 else f[1].format(x) for (n,x) in enumerate(a)]
        self.xticklabels = ['\n'.join(('missing','(nan)'))] + labels
        if max(a[1:]**2) >= 1e6: self.rotation = 45
        else: self.rotation = 0

    def __woe_plot(self, axis):
        
        title = ['Variable: %s' % self.var_name, 
                 'IV = %.4f (%s)' % (self.iv, self.__iv_predict())]

        # extract positive and negative woes
        a = self.df['woe'].values
        pos = np.array([[n,w] for n,w in enumerate(a) if w>=0])
        neg = np.array([[n,w] for n,w in enumerate(a) if w<0])
        
        # plot woes
        axis.bar(pos[:,0], pos[:,1], **self.kw['p_woe'])
        axis.bar(neg[:,0], neg[:,1], **self.kw['n_woe'])
        
        # woe values (text)
        for p in pos:
            axis.text(p[0], -0.05, '%0.2f'%p[1], **self.kw['p_text'])
        for n in neg:
            axis.text(n[0], +0.05, '%0.2f'%n[1], **self.kw['n_text'])
        
        axis.set_facecolor('white')
        axis.set_ylabel('Weight of Evidence (WOE)')
        axis.set_xticks(np.arange(len(a)))
        kwargs = dict(fontsize=10, rotation=self.rotation)
        axis.set_xticklabels(self.xticklabels, **kwargs)
        axis.set_title('\n'.join(tuple(title)))
        
        # set y ticks
        yticks = axis.get_yticks()
        diff = max(np.diff(yticks))
        n_low, n_high = min(yticks)-diff, max(yticks)+diff
        axis.set_ylim(n_low, n_high)
        
        s = r'$\rho_{data}$ = %s'%(self.rho)
        bbx_kw = dict(boxstyle='round',facecolor='white',alpha=0)
        axis.text(0.02, 0.02, s, transform=axis.transAxes, fontsize=10, 
                  va='bottom', ha='left', bbox=bbx_kw) 
  
        # legend
        axis.legend(**self.kw['legend'])
        axis.grid(False)

    def __distribution_plot(self, axis):

        title = 'Variable: %s \n samples (%%) in each BIN' % self.var_name
        pct = self.df[['pct_nonevents','pct_events']].values*100
      
        # plot distribution
        x = np.arange(len(pct))
        axis.bar(x, +pct[:,0], **self.kw['non_event'])
        axis.bar(x, -pct[:,1], **self.kw['event'])

        # distribution percentage with respect to group (text)
        for n, s in enumerate(pct):
            axis.text(n, +s[0]+1, '%d%%'%s[0], **self.kw['n_text'])
            axis.text(n, -s[1]-1, '%d%%'%s[1], **self.kw['p_text'])

        axis.set_facecolor('white')
        axis.set_ylabel('Percentage (%)')
        axis.set_xticks(x)
        kwargs = dict(fontsize=10, rotation=self.rotation)
        axis.set_xticklabels(self.xticklabels, **kwargs)
        axis.set_title(title)
        
        # set y ticks
        yticks = axis.get_yticks()
        diff = max(np.diff(yticks))
        low = max(min(yticks)-diff,-100)
        high = min(max(yticks)+diff,100)
        axis.set_ylim(low, high)
        
        # legend
        axis.legend(**self.kw['legend'])
        axis.grid(False)
             
    def __target_rate_plot(self, axis):
        
        title = 'Variable: %s \n Target Rate (%%) in each BIN' % self.var_name
        n_sample = np.sum(self.df[['non_events','events']],axis=1)
        pct_sample = n_sample/sum(n_sample)*100
        pct_bad = self.df['events'].values/n_sample*100
          
        # plot distribution
        x = np.arange(len(n_sample))
        tw_axis = axis.twinx()
        bar = axis.bar(x, pct_sample,**self.kw['sample'])
        line = tw_axis.plot(x, pct_bad, **self.kw['target'])
        plots, labels = [bar,line[0]], [bar.get_label(),line[0].get_label()]
        
        for n, s in enumerate(pct_sample):
            axis.text(n, s+0.5, '%d%%'%s, **self.kw['n_text'])
           
        axis.set_facecolor('white')
        axis.set_ylabel('Percentage (%)')
        tw_axis.set_ylabel('Target Rate (%)')
        axis.set_xticks(x)
        kwargs = dict(fontsize=10, rotation=self.rotation)
        axis.set_xticklabels(self.xticklabels, **kwargs)
        axis.set_title(title)
        
        # set y ticks
        yticks = axis.get_yticks()
        diff = max(np.diff(yticks))
        axis.set_ylim(0, min(max(yticks)+diff,100))
        
        # legend
        axis.legend(plots,labels,**self.kw['legend'])
        axis.grid(False)
        
class woe_transform:
      
    '''
    Parameters
    ----------
    woe_df : dataframe object with mandatory fields
    (1) 'variable': name of variables
    (2) 'min','max': BIN intervals (min <= x < max)
    (3) 'bin': BIN number
    (4) 'woe': Weight of Evidence

    Returns
    -------
    self.woe : dataframe object (WOE table)
   
    Method
    ------
    self.fit(X)
    \t Fit the model according to the given initial inputs
    '''

    def __init__(self, woe_df):
        self.woe = woe_df.rename(str.lower,axis=1)

    def fit(self, X):

        '''
        Transform variables into WOEs (Weight of Evidence) according 
        to the given bin intervals. This method only transforms 
        variables that only exist in woe_df (WOE table).
        
        Parameters
        ----------
        X : dataframe object, shape of (n_samples, n_features)
        \t Dataframe that will be tranformed into WOEs to fit the model
        
        Returns
        -------
        self.X : dataframe object
        \t Transformed dataframe with all WOEs.
        
        Note: missing or np.nan is binned separately
        '''
        if isinstance(X, tuple((pd.core.series.Series, pd.DataFrame))): 
            woe_var = np.unique(self.woe['variable'].values)
            columns = [var for var in X.columns if var in woe_var]
            if len(columns) > 0:
                a = [self.__assign_woe(X[var]) for var in columns]
                self.X = pd.DataFrame(np.hstack(a),columns=columns)
            else: print('Variables from input do not match with WOE table !!!')
        else: print('Input must be either Series or Dataframe !!!')
                    
    def __assign_woe(self, X):

        # determine min and max
        woe_df = self.woe_df.loc[(self.woe_df['variable']==X.name)]
        nan_woe = float(woe_df.loc[woe_df['bin']==0,'woe'].values)
        bin_edges = (woe_df['min'].values.tolist() + [float('Inf')])[1:]

        # Assign group index to value array and convert to dataframe
        X = np.digitize(X, bin_edges, right=False)
        X = pd.DataFrame(data=X, columns=['bin'])
        X.loc[X['bin']==len(bin_edges),'bin'] = 0
        X = X.merge(woe_df[['bin','woe']], on=['bin'], how='left')
        return X['woe'].values.reshape(-1,1)
