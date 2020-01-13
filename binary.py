import pandas as pd, numpy as np, math, time, os
from matplotlib import cm
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import ipywidgets as widgets

class binary_analysis:
    
    '''   
    Methods
    -------
    
    self.fit(X, y, min_pct=0.01, bool_pct=0.9)
    \t determine binary combinations of all variables
    
    self.plot(var, figsize=(12,8), fname=None):
    \t plot the combination result with selected starting variable
    
    self.plot_sum(min_lift=1, n_var=10, top=5, figsize=(12,4.5), use_cmap=True, fname=None)
    \t plot combination(s) that passes predefined criteria
    '''
    def __init__(self, lift_df=None, decile_c='#c23616', cumu_c='#0097e6', 
                 sample_c='#c23616', target_c='#0097e6', cmap='cool'):
        
        '''
        Parameters
        ----------
        
        lift_df : dataframe, optional, default: None
        \t see example of dataframe in self.fit().__doc__
        
        decile_c : hex, optional, default: '#c23616'
        \t color of PER-DECILE line plot
        
        cumu_c : hex, optional, default: '#0097e6'
        \t color of CUMULATIVE line plot
        
        sample_c : hex, optional, default: '#c23616'
        \t color of SAMPLE line plot
        
        target_c : hex, optional, default: '#0097e6'
        \t color of TARGET line plot
        
        cmap : str, optional, default: 'cool'
        \t name of matplotlib.cm (color map)
        '''
        self.lift_df = lift_df
        self.marker = dict(lw=1, ls='-', marker='o', markersize=5, markerfacecolor='none')
        self.c_lift = self.marker.copy(); self.c_lift.update(dict(color=cumu_c, label='Cumulative Lift'))     
        self.d_lift = self.marker.copy(); self.d_lift.update(dict(color=decile_c, label='Per-decile Lift'))
        self.sample = self.marker.copy(); self.sample.update(dict(color=sample_c, label='Cum. Sample'))     
        self.target = self.marker.copy(); self.target.update(dict(color=target_c, label='Cum. Target'))
        self.legend1 = dict(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0, edgecolor='none')
        self.legend2 = dict(loc='best', fontsize=10, framealpha=0, edgecolor='none')
        self.v_span = dict(ymin=0, ymax=1, color='#718093', alpha=0.1, hatch='///', label=r'Per-decile > 0')
        self.t_display = dict(color='k',fontsize=10, va='top', bbox=dict(facecolor='none',edgecolor='none'))
        self.cmap = cmap
       
    def __lift(self, features):

        ''' 
        index_np is an array of 0 and 1 
        a = no. of targets, b = no. of acquired samples
        '''
        flag = np.sum(self.X[features],axis=1).astype(bool)
        a, b = len(self.y[(flag==1)&(self.y==1)]), (flag==1).sum()
        lift = (a/self.n_event)/float(b/self.n_samples)
        return a, b, lift
  
    def __best_feature(self, features):

        '''
        Through iteration process, this function determines 
        variable that provides the highest incremental lift
        '''
        if isinstance(features,str): features = [features]
        combi, by_decile, new_event, new_nonevent = None, 0, 0, 0
        n_event, n_sample, _ = self.__lift(features)
        rm_features = set(self.features).difference(features)
        
        for var in rm_features:
            
            # change of event over change of non-event (lift @ decile)
            a, b, _ = self.__lift(features + [var])
            chg_event = (a-n_event)/self.n_event
            chg_sample = (b-n_sample)/self.n_samples
            if chg_sample > 0: chg = chg_event/chg_sample
            else: chg = 0
            # optimize lift-per-decile
            if (by_decile<chg) & (chg>0):
                combi, by_decile = features + [var], chg
                new_event, new_nonevent = a, b
        return combi, by_decile, new_event, new_nonevent
  
    def __create_df(self, feature):

        df = pd.DataFrame(columns=self.columns)
        a, b, lift = self.__lift([feature])
        c = dict(combi_var=feature, target=a, sample=b, d_lift=lift)
        return df.append(c, ignore_index=True).reset_index(drop=True)

    def __find_feature(self, feature):

        '''
        Find the subsequent feature from remaining 
        features that provides the best lift
        '''
        df, new_feature = self.__create_df(feature), feature
        while new_feature != None:
            new_feature, lift, a, b = self.__best_feature(new_feature)
            if lift > 0:
                self.w_t2.value = ' ==> no. of variables = {:.0f}'.format(len(new_feature))
                a = dict(combi_var=new_feature[-1], target=a, sample=b, d_lift=lift)
                df = df.append(a,ignore_index=True).reset_index(drop=True)
        df['pct_target'] = df['target']/self.n_event*100
        df['pct_sample'] = df['sample']/self.n_samples*100
        df['c_lift'] = df['pct_target']/df['pct_sample']
        df['variable'] = feature
        return df
  
    def fit(self, X, y, min_pct=0.01, bool_pct=0.9):
        
        '''
        * * Screening Criteria * *
        (1) X must be boolean, X={0,1}, and all missing is replaced with 0
        (2) count(y==1|flag==0) < count(y==1) * XX%
        (3) sum of flags <= YY%
        
        Parameters
        ----------
        
        X : array-like shape of (n_samples , n_features)
        \t binary variables
        
        y : array-like of shape (n_samples,)
        \t target values (binary)
        
        min_pct : float, optional, default: 0.01 (1%)
        \t minimum number of targets given flag==1
        
        bool_pct : float, optional, default: 0.9 (90%)
        \t maximum percentage of True in variable
        
        Return
        ------
        
        self.lift_df : dataframe
        ======================================================================================
        | variable | combi_var | target | sample | d_lift | pct_target | pct_sample | c_lift |
        --------------------------------------------------------------------------------------
        |   XXXX   |   XXXX    |   286  |  2605  |  5.17  |    63.55   |   12.27    |  5.17  |
        |   XXXX   |   AAAA    |   310  |  2846  |  4.69  |    68.88   |   13.40    |  5.13  |
        |   XXXX   |   BBBB    |   316  |  2920  |  3.82  |    70.22   |   13.75    |  5.10  |
        |   XXXX   |   CCCC    |   322  |  2997  |  3.67  |    71.55   |   14.12    |  5.06  |
        ======================================================================================
        Note: combi_var = 'combination of variables', d_lift = 'lift by decile', and c_lift = 'cumulative lift'
        '''
        # start time
        start = time.time()
        
        # 1st Condition: X must be boolean = {0,1}
        Xs = X.iloc[:,((np.nanmax(X,axis=0)==1) & (np.nanmin(X,axis=0)==0))].fillna(0).copy()
        
        # 2nd Condition: y==1 and flag==0
        min_flag = min_pct * sum(y)
        Xs = Xs.iloc[:,(Xs.loc[(y==1),:].sum(axis=0)>min_flag).values]

        # 3rd Condition: sum of flag <= XX%
        Xs = Xs.iloc[:,((Xs.sum(axis=0)/len(Xs))<=bool_pct).values]
        
        # Features that satisfies above requirements
        self.features, self.y, self.X = list(Xs), y.copy(), Xs.copy()
        self.n_samples, self.n_event, self.n_nonevent = len(y), sum(y), (y==0).sum()
        
        # initial variables
        self.__widgets()
        self.columns = ['variable','combi_var','target','sample','d_lift','pct_target','pct_sample','c_lift']
        df, n_features = pd.DataFrame(columns=self.columns), len(self.features)
        
        for (n,var) in enumerate(self.features,1):
            self.w_t1.value =  ' ' + var + ' ( {:.0f}% )'.format((n/n_features)*100)
            df = df.append(self.__find_feature(var),ignore_index=True).reset_index(drop=True)
        df.loc[:,self.columns[2:]] = df.loc[:,self.columns[2:]].astype('float64')
        self.lift_df = df.copy()
        self.w_t1.value = 'Complete'
        self.w_t2.value = ' ==> Number of (bool) features : %d' % n_features
        print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
            
    def __widgets(self):

        self.w_t1 = widgets.HTMLMath(value='Calculating . . .')
        self.w_t2 = widgets.HTMLMath(value='')
        w = widgets.HBox([self.w_t1, self.w_t2])
        display(w); time.sleep(2)
        
    def __cumu_plot(self, axis, var, loc=None):
        
        '''
        Cumulative plot of samples and tartget
        '''
        # ticks and labels
        a = self.lift_df[self.lift_df['variable']==str(var)].reset_index(drop=True).copy()
        xticklabels = np.array(a['combi_var']).tolist()
        xticklabels = np.arange(1,len(xticklabels)+1)
        xticks = np.arange(len(xticklabels))

        # Primary axis
        ln1 = axis.plot(xticks, a['pct_sample'], **self.sample)
        axis.set_ylabel(r'Cumulative no. of $\bf{Samples}$ (%)',color=self.sample['color'])
        axis.set_facecolor('white')
        axis.set_xlim(-0.5,len(xticks)-0.5)
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels, fontsize=10)
        axis.set_xlabel(r'$\bf{Variables}$')
        title = (r'$\bf{BINARY}$ $\bf{COMBINATION}$ $\bf{ANALYSIS}$','Variable : %s' % var)
        axis.set_title('\n'.join(tuple(title)), fontsize=12)
        axis.grid(False)

        # Secondary axis
        tw_axis = axis.twinx()
        ln2 = tw_axis.plot(xticks, a['pct_target'], **self.target)
        tw_axis.set_ylabel(r'Cumulative no. of $\bf{Targets}$ (%)',color=self.target['color'])
        tw_axis.grid(False)
        
        # cutoff line
        n_cutoff = np.argmax(abs(a['pct_sample'].values-a['pct_target'].values))
        axis.axvline(xticks[n_cutoff], color='k', linestyle="--", lw=0.8)
        kwargs = self.v_span.copy(); kwargs['label'] = 'Max. Gap (cutoff)'
        sp1 = axis.axvspan(-0.5,xticks[n_cutoff],**kwargs)
        axis.legend([ln1[0],ln2[0],sp1],[ln1[0].get_label(),ln2[0].get_label(),sp1.get_label()],**self.legend1)
        
        # display
        s, t, c = list(), list(), list()
        s.append('%s (%d%%)' % ('{:,.0f}'.format(a['target'].min()), a['pct_target'].min()))
        s.append('%s (%d%%)' % ('{:,.0f}'.format(a['target'].max()), a['pct_target'].max()))
        t.append('%s (%d%%)' % ('{:,.0f}'.format(a['sample'].min()), a['pct_sample'].min()))
        t.append('%s (%d%%)' % ('{:,.0f}'.format(a['sample'].max()), a['pct_sample'].max()))
        c.append('{T=%s (%d%%)' % ('{:,.0f}'.format(a.loc[n_cutoff,'target']), a.loc[n_cutoff,'pct_target']))
        c.append('S=%s (%d%%)}' % ('{:,.0f}'.format(a.loc[n_cutoff,'sample']), a.loc[n_cutoff,'pct_sample']))
        b = tuple(('Max. Combination = %d' % len(xticks),
                   'Target = ' + ' , '.join(tuple(t)),
                   'Sample = ' + ' , '.join(tuple(s)),
                   'Cutoff Combination = %d ' % (n_cutoff+1),
                   r'Cutoff : ' + ' , '.join(tuple(c))))
        self.t_display.update(dict(transform=axis.transAxes))
        axis.text(0.02, 0.95, '\n'.join(b),**self.t_display)
        
    def __lift_plot(self, axis, var):
        
        '''
        Lift plots i.e. by-decile and cumulative (normal)
        '''
        a = self.lift_df[self.lift_df['variable']==str(var)].copy()
        xticklabels = np.array(a['combi_var']).tolist()
        xticklabels = np.arange(1,len(xticklabels)+1)
        xticks = np.arange(len(xticklabels))
        
        # plot
        line_1 = axis.plot(xticks, a['d_lift'], **self.d_lift)
        line_2 = axis.plot(xticks, a['c_lift'], **self.c_lift)
        axis.set_facecolor('white')
        axis.set_ylabel(r'$\bf{Lift}$')
        axis.set_xlabel(r'$\bf{Variables}$')
        axis.set_xticks(xticks)
        axis.set_xlim(-0.5,len(xticks)-0.5)
        axis.set_xticklabels(xticklabels, fontsize=10)
        title = (r'$\bf{LIFT}$ $\bf{CHART}$','Cumulative and Per-decile')
        axis.set_title('\n'.join(tuple(title)), fontsize=12)
        axis.grid(False)
        axis.axhline(1, color='k', linestyle="--", lw=0.8)
        
        # determine position where lift per decile >= 1
        n = sum(np.array((a['d_lift'].values>= 1).astype(int)))
        axis.axvline(xticks[n-1], color='k', linestyle="--", lw=0.8)
        axis.axvspan(-0.5,xticks[n-1],**self.v_span)
        xy = (xticks[n-1]-0.1,axis.get_ylim()[1]*0.98)
        axis.annotate(r'Combination = $\bf{%d}$ ' % (n), xy, va='top', ha='right', fontsize=10)
        axis.legend(**self.legend1)
        
    def __var_table(self, axis, var):
        
        cellHeight = 0.045
        a = self.lift_df.loc[self.lift_df['variable']==var,['combi_var']].reset_index()
        a.iloc[:,0] = np.arange(1,len(a)+1)
        
        axis.yaxis.set_visible(False)
        axis.xaxis.set_visible(False)
        axis.set_axis_off()
        for pos in ['right','top','left','bottom']:
            axis.spines[pos].set_visible(False)
        axis.set_title(r'$\bf{LIST}$ $\bf{OF}$ $\bf{VARIABLES}$')
        
        tb_length = min(cellHeight*len(a),1)
        axis.table(cellText=a.values, colLabels=['Order','Variable'], cellLoc = 'left', 
                   rowLoc = 'left', loc='upper left', bbox=[0,1-tb_length,1,tb_length], 
                   fontsize=10, colWidths=[0.2,0.8],
                   cellColours=[['0.95' for c in range(2)] for r in range(len(a))])
    
    def __bold_txt(self, a, ucase=False, pat=' '):
        
        '''
        Parameters
        ----------
        
        a : string
        \t string with any length
        
        ucase : boolean, optional (default=False)
        \t when True, string is capitalized
        
        pat : str, optional (default=' ')
        \t string or regular expression to split on. If not specified, split on whitespace.
        '''
        return tuple([r'$\bf{s}$'.format(s=s) if ucase==False 
                      else r'$\bf{s}$'.format(s=s.upper()) for s in str(a).split(pat)])
        
    def plot(self, var, figsize=(12,8), fname=None):
    
        '''
        Parameters
        ----------
        
        var : str
        \t variable name
        
        figsize : (float,float), optional, default: (12,8)
        \t width, height in inches of plot
        
        fname : str or PathLike or file-like object
        \t A path, or a Python file-like object (see pyplot.savefig)
        '''
        fig = plt.figure(figsize=figsize)
        axis = np.full(3,None)
        axis[0] = plt.subplot2grid((2,3),(0,0), colspan=2)
        axis[1] = plt.subplot2grid((2,3),(1,0), colspan=2)
        axis[2] = plt.subplot2grid((2,3),(0,2), rowspan=2)
        self.__cumu_plot(axis[0], var)
        self.__lift_plot(axis[1], var)
        self.__var_table(axis[2], var)
        fig.tight_layout()
        if fname != None: plt.savefig(fname)
        plt.show()
        
    def __filter_vars(self, min_lift=1, max_var=10, top=10):
        
        features = np.unique(self.lift_df['variable'].values).tolist()
        columns = ['variable','c_lift','pct_target','pct_sample']
        b = pd.DataFrame(columns=columns)
        for var in features:
            a = self.lift_df[self.lift_df['variable']==var].reset_index(drop=True)
            a = a.loc[(a['d_lift']>=min_lift) & (a.index<=max_var-1), columns]
            b = b.append(a,ignore_index=True).reset_index(drop=True)
        
        # determine top n features based on cumulative lift
        m_features = b.groupby(['variable']).agg('max')['c_lift']
        m_features = np.array(m_features.sort_values(0,ascending=False).index)[:top]
        return b.loc[b['variable'].isin(m_features)].reset_index(drop=True), m_features

    def plot_sum(self, min_lift=1, n_var=10, top=5, figsize=(12,4.5), use_cmap=True, fname=None):
        
        '''
        Parameters
        ----------
        
        min_lift : float, optional, default: 1
        \t minimum per-decile lift threshold.
        
        n_var : int, optional, default: 10
        \t maximum number of combined variables
        
        top : int, optional, default: 5
        \t ranking within predefined n ranks
        
        figsize : (float, float), optional, default: (12,4.5)
        \t width, height in inches of plot
        
        use_cmap : boolean, optional, default: True
        \t if True, plot will use predefined color map from matplotlib.
        \t However, if False, plot will use defualt palette.
        
        fname : str or PathLike or file-like object
        \t A path, or a Python file-like object (see pyplot.savefig)
        '''
        gridsize, axis = (1,3), np.full(3,None)
        fig = plt.figure(figsize=figsize)
        axis[0] = plt.subplot2grid(gridsize,(0,0))
        axis[1] = plt.subplot2grid(gridsize,(0,1))
        axis[2] = plt.subplot2grid(gridsize,(0,2))
        var = ['c_lift','pct_target','pct_sample']
        title = [r'$\bf{Lift}$ $\bf{Chart}$',r'$\bf{Target}$',r'$\bf{Sample}$']
        y_label = [r'$\bf{Lift}$',r'$\bf{Cumulative}$ $\bf{Percentage}$ (%)']
        
        a, m_var = self.__filter_vars(min_lift, n_var, top)
        args = [(axis[0], a[['variable',var[0]]], y_label[0], title[0]),
                (axis[1], a[['variable',var[1]]], y_label[1], title[1]),
                (axis[2], a[['variable',var[2]]], y_label[1], title[2])]
        for n in range(3):
            self.__plot_chart(*(args[n]+(n_var, m_var, use_cmap)))
        fig.tight_layout()
        if fname != None: plt.savefig(fname)
        plt.show()
    
    def __plot_chart(self, axis, a, ylabel, title, n_var, m_var, use_cmap):
        
        cmp = self.__cmap(len(m_var))
        kwargs = self.marker.copy()
        for (n,var) in enumerate(m_var,1):
            y = a.loc[a['variable']==var].values[:,1]
            kwargs['label'] = '(%d) %s' % (n,var)
            if use_cmap: kwargs['color'] = cmp[n-1]
            axis.plot(range(len(y)), y, **kwargs)
        axis.set_facecolor('white')
        axis.set_ylabel(ylabel, fontsize=10)
        axis.set_xlabel(r'$\bf{Number}$ $\bf{of}$ $\bf{Variables}$', fontsize=10)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(range(n_var))
        axis.set_xticklabels(np.arange(1,n_var+1), fontsize=10)
        axis.legend(**self.legend2)
        axis.grid(False)
        
    def __cmap(self, n_color=10):
        
        cmap = cm.get_cmap(self.cmap)(np.linspace(0,1,n_color))
        cmap = (cmap*255).astype(int)[:,:3]
        cmap =['#%02x%02x%02x' % (cmap[i,0],cmap[i,1],cmap[i,2]) for i in range(n_color)]
        return cmap
## Version2.0 ##
class binary_analysis2:
    
    '''   
    Methods
    -------
    
    self.fit(X, y, min_pct=0.01, bool_pct=0.9)
    \t determine binary combinations of all variables
    
    self.plot(var, figsize=(12,8), fname=None):
    \t plot the combination result with selected starting variable
    
    self.plot_sum(min_lift=1, n_var=10, top=5, figsize=(12,4.5), use_cmap=True, fname=None)
    \t plot combination(s) that passes predefined criteria
    '''
    def __init__(self, lift_df=None, decile_c='#c23616', cumu_c='#0097e6', 
                 sample_c='#c23616', target_c='#0097e6', cmap='cool'):
        
        '''
        Parameters
        ----------
        
        lift_df : dataframe, optional, default: None
        \t see example of dataframe in self.fit().__doc__
        
        decile_c : hex, optional, default: '#c23616'
        \t color of PER-DECILE line plot
        
        cumu_c : hex, optional, default: '#0097e6'
        \t color of CUMULATIVE line plot
        
        sample_c : hex, optional, default: '#c23616'
        \t color of SAMPLE line plot
        
        target_c : hex, optional, default: '#0097e6'
        \t color of TARGET line plot
        
        cmap : str, optional, default: 'cool'
        \t name of matplotlib.cm (color map)
        '''
        self.lift_df = lift_df
        self.marker = dict(lw=1, ls='-', marker='o', markersize=5, markerfacecolor='none')
        self.c_lift = self.marker.copy(); self.c_lift.update(dict(color=cumu_c, label='Cumulative Lift'))     
        self.d_lift = self.marker.copy(); self.d_lift.update(dict(color=decile_c, label='Per-decile Lift'))
        self.sample = self.marker.copy(); self.sample.update(dict(color=sample_c, label='Cum. Sample'))     
        self.target = self.marker.copy(); self.target.update(dict(color=target_c, label='Cum. Target'))
        self.legend1 = dict(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0, edgecolor='none')
        self.legend2 = dict(loc='best', fontsize=10, framealpha=0, edgecolor='none')
        self.v_span = dict(ymin=0, ymax=1, color='#718093', alpha=0.1, hatch='///', label=r'Per-decile > 0')
        self.t_display = dict(color='k',fontsize=10, va='top', bbox=dict(facecolor='none',edgecolor='none'))
        self.cmap = cmap
  
    def fit(self, X, y, min_pct=0.01, bool_pct=0.9, min_clift=1, min_dlift=0):
        
        '''
        * * Screening Criteria * *
        (1) X must be boolean, X={0,1}, and all missing is replaced with 0
        (2) count(y==1|flag==0) < count(y==1) * XX%
        (3) sum of flags <= YY%
        
        Parameters
        ----------
        
        X : array-like shape of (n_samples , n_features)
        \t binary variables
        
        y : array-like of shape (n_samples,)
        \t target values (binary)
        
        min_pct : float, optional, default: 0.01 (1%)
        \t minimum number of targets given flag==1
        
        bool_pct : float, optional, default: 0.9 (90%)
        \t maximum percentage of True in variable
        
        min_clift : float, optional, default: 1
        \t the minimum cumulative lift required to continue.

        min_dlift : float, optional, default: 0
        \t the minimum per-decile lift required to continue.
        
        Return
        ------
        
        self.lift_df : dataframe
        ======================================================================================
        | variable | combi_var | target | sample | d_lift | pct_target | pct_sample | c_lift |
        --------------------------------------------------------------------------------------
        |   XXXX   |   XXXX    |   286  |  2605  |  5.17  |    63.55   |   12.27    |  5.17  |
        |   XXXX   |   AAAA    |   310  |  2846  |  4.69  |    68.88   |   13.40    |  5.13  |
        |   XXXX   |   BBBB    |   316  |  2920  |  3.82  |    70.22   |   13.75    |  5.10  |
        |   XXXX   |   CCCC    |   322  |  2997  |  3.67  |    71.55   |   14.12    |  5.06  |
        ======================================================================================
        Note: combi_var = 'combination of variables', d_lift = 'lift by decile', and c_lift = 'cumulative lift'
        '''
        # start time
        start = time.time()
        
        # 1st Condition: X must be boolean = {0,1}
        Xs = X.iloc[:,((np.nanmax(X,axis=0)==1) & (np.nanmin(X,axis=0)==0))].fillna(0).copy()
        
        # 2nd Condition: y==1 and flag==0
        min_flag = min_pct * sum(y)
        Xs = Xs.iloc[:,(Xs.loc[(y==1),:].sum(axis=0)>min_flag).values]

        # 3rd Condition: sum of flag <= XX%
        Xs = Xs.iloc[:,((Xs.sum(axis=0)/len(Xs))<=bool_pct).values]
        
        # Features that satisfies above requirements
        self.features, self.y, self.X = list(Xs), y.copy(), Xs.copy()
        self.n_samples, self.n_event, self.n_nonevent = len(y), sum(y), (y==0).sum()
        
        # initial variables
        self.__widgets()
        self.columns = ['variable','combi_var','target','sample','d_lift','pct_target','pct_sample','c_lift']
        df, n_features = pd.DataFrame(columns=self.columns), len(self.features)
        
        for (n,var) in enumerate(self.features,1):
            self.w_t1.value =  ' ' + var + ' ( {:.0f}% )'.format((n/n_features)*100)
            kwargs = dict(c_features=[var], min_clift=min_clift, min_dlift=min_dlift)
            a = [np.array(n).reshape(-1,1) for n in seq_lift(self.X, self.y, **kwargs)]
            a = [np.full((len(a[0]),1),var)] + a
            a = pd.DataFrame(np.hstack(a), columns=self.columns)
            df = df.append(a ,ignore_index=True).reset_index(drop=True)
        df.loc[:,self.columns[2:]] = df.loc[:,self.columns[2:]].astype('float64')
        self.lift_df = df.copy()
        self.w_t1.value = 'Complete'
        self.w_t2.value = ' ==> Number of (bool) features : %d' % n_features
        print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
            
    def __widgets(self):

        self.w_t1 = widgets.HTMLMath(value='Calculating . . .')
        self.w_t2 = widgets.HTMLMath(value='')
        w = widgets.HBox([self.w_t1, self.w_t2])
        display(w); time.sleep(2)
        
    def __cumu_plot(self, axis, var, loc=None):
        
        '''
        Cumulative plot of samples and tartget
        '''
        # ticks and labels
        a = self.lift_df[self.lift_df['variable']==str(var)].reset_index(drop=True).copy()
        xticklabels = np.array(a['combi_var']).tolist()
        xticklabels = np.arange(1,len(xticklabels)+1)
        xticks = np.arange(len(xticklabels))

        # Primary axis
        ln1 = axis.plot(xticks, a['pct_sample'], **self.sample)
        axis.set_ylabel(r'Cumulative no. of $\bf{Samples}$ (%)',color=self.sample['color'])
        axis.set_facecolor('white')
        axis.set_xlim(-0.5,len(xticks)-0.5)
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels, fontsize=10)
        axis.set_xlabel(r'$\bf{Variables}$')
        title = (r'$\bf{BINARY}$ $\bf{COMBINATION}$ $\bf{ANALYSIS}$','Variable : %s' % var)
        axis.set_title('\n'.join(tuple(title)), fontsize=12)
        axis.grid(False)

        # Secondary axis
        tw_axis = axis.twinx()
        ln2 = tw_axis.plot(xticks, a['pct_target'], **self.target)
        tw_axis.set_ylabel(r'Cumulative no. of $\bf{Targets}$ (%)',color=self.target['color'])
        tw_axis.grid(False)
        
        # cutoff line
        n_cutoff = np.argmax(abs(a['pct_sample'].values-a['pct_target'].values))
        axis.axvline(xticks[n_cutoff], color='k', linestyle="--", lw=0.8)
        kwargs = self.v_span.copy(); kwargs['label'] = 'Max. Gap (cutoff)'
        sp1 = axis.axvspan(-0.5,xticks[n_cutoff],**kwargs)
        axis.legend([ln1[0],ln2[0],sp1],[ln1[0].get_label(),ln2[0].get_label(),sp1.get_label()],**self.legend1)
        
        # display
        s, t, c = list(), list(), list()
        s.append('%s (%d%%)' % ('{:,.0f}'.format(a['target'].min()), a['pct_target'].min()))
        s.append('%s (%d%%)' % ('{:,.0f}'.format(a['target'].max()), a['pct_target'].max()))
        t.append('%s (%d%%)' % ('{:,.0f}'.format(a['sample'].min()), a['pct_sample'].min()))
        t.append('%s (%d%%)' % ('{:,.0f}'.format(a['sample'].max()), a['pct_sample'].max()))
        c.append('{T=%s (%d%%)' % ('{:,.0f}'.format(a.loc[n_cutoff,'target']), a.loc[n_cutoff,'pct_target']))
        c.append('S=%s (%d%%)}' % ('{:,.0f}'.format(a.loc[n_cutoff,'sample']), a.loc[n_cutoff,'pct_sample']))
        b = tuple(('Max. Combination = %d' % len(xticks),
                   'Target = ' + ' , '.join(tuple(t)),
                   'Sample = ' + ' , '.join(tuple(s)),
                   'Cutoff Combination = %d ' % (n_cutoff+1),
                   r'Cutoff : ' + ' , '.join(tuple(c))))
        self.t_display.update(dict(transform=axis.transAxes))
        axis.text(0.02, 0.95, '\n'.join(b),**self.t_display)
        
    def __lift_plot(self, axis, var):
        
        '''
        Lift plots i.e. by-decile and cumulative (normal)
        '''
        a = self.lift_df[self.lift_df['variable']==str(var)].copy()
        xticklabels = np.array(a['combi_var']).tolist()
        xticklabels = np.arange(1,len(xticklabels)+1)
        xticks = np.arange(len(xticklabels))
        
        # plot
        line_1 = axis.plot(xticks, a['d_lift'], **self.d_lift)
        line_2 = axis.plot(xticks, a['c_lift'], **self.c_lift)
        axis.set_facecolor('white')
        axis.set_ylabel(r'$\bf{Lift}$')
        axis.set_xlabel(r'$\bf{Variables}$')
        axis.set_xticks(xticks)
        axis.set_xlim(-0.5,len(xticks)-0.5)
        axis.set_xticklabels(xticklabels, fontsize=10)
        title = (r'$\bf{LIFT}$ $\bf{CHART}$','Cumulative and Per-decile')
        axis.set_title('\n'.join(tuple(title)), fontsize=12)
        axis.grid(False)
        axis.axhline(1, color='k', linestyle="--", lw=0.8)
        
        # determine position where lift per decile >= 1
        n = sum(np.array((a['d_lift'].values>= 1).astype(int)))
        axis.axvline(xticks[n-1], color='k', linestyle="--", lw=0.8)
        axis.axvspan(-0.5,xticks[n-1],**self.v_span)
        xy = (xticks[n-1]-0.1,axis.get_ylim()[1]*0.98)
        axis.annotate(r'Combination = $\bf{%d}$ ' % (n), xy, va='top', ha='right', fontsize=10)
        axis.legend(**self.legend1)
        
    def __var_table(self, axis, var):
        
        cellHeight = 0.045
        a = self.lift_df.loc[self.lift_df['variable']==var,['combi_var']].reset_index()
        a.iloc[:,0] = np.arange(1,len(a)+1)
        
        axis.yaxis.set_visible(False)
        axis.xaxis.set_visible(False)
        axis.set_axis_off()
        for pos in ['right','top','left','bottom']:
            axis.spines[pos].set_visible(False)
        axis.set_title(r'$\bf{LIST}$ $\bf{OF}$ $\bf{VARIABLES}$')
        
        tb_length = min(cellHeight*len(a),1)
        axis.table(cellText=a.values, colLabels=['Order','Variable'], cellLoc = 'left', 
                   rowLoc = 'left', loc='upper left', bbox=[0,1-tb_length,1,tb_length], 
                   fontsize=10, colWidths=[0.2,0.8],
                   cellColours=[['0.95' for c in range(2)] for r in range(len(a))])
    
    def __bold_txt(self, a, ucase=False, pat=' '):
        
        '''
        Parameters
        ----------
        
        a : string
        \t string with any length
        
        ucase : boolean, optional (default=False)
        \t when True, string is capitalized
        
        pat : str, optional (default=' ')
        \t string or regular expression to split on. If not specified, split on whitespace.
        '''
        return tuple([r'$\bf{s}$'.format(s=s) if ucase==False 
                      else r'$\bf{s}$'.format(s=s.upper()) for s in str(a).split(pat)])
        
    def plot(self, var, figsize=(12,8), fname=None):
    
        '''
        Parameters
        ----------
        
        var : str
        \t variable name
        
        figsize : (float,float), optional, default: (12,8)
        \t width, height in inches of plot
        
        fname : str or PathLike or file-like object
        \t A path, or a Python file-like object (see pyplot.savefig)
        '''
        fig = plt.figure(figsize=figsize)
        axis = np.full(3,None)
        axis[0] = plt.subplot2grid((2,3),(0,0), colspan=2)
        axis[1] = plt.subplot2grid((2,3),(1,0), colspan=2)
        axis[2] = plt.subplot2grid((2,3),(0,2), rowspan=2)
        self.__cumu_plot(axis[0], var)
        self.__lift_plot(axis[1], var)
        self.__var_table(axis[2], var)
        fig.tight_layout()
        if fname != None: plt.savefig(fname)
        plt.show()
        
    def __filter_vars(self, min_lift=1, max_var=10, top=10):
        
        features = np.unique(self.lift_df['variable'].values).tolist()
        columns = ['variable','c_lift','pct_target','pct_sample']
        b = pd.DataFrame(columns=columns)
        for var in features:
            a = self.lift_df[self.lift_df['variable']==var].reset_index(drop=True)
            a = a.loc[(a['d_lift']>=min_lift) & (a.index<=max_var-1), columns]
            b = b.append(a,ignore_index=True).reset_index(drop=True)
        
        # determine top n features based on cumulative lift
        m_features = b.groupby(['variable']).agg('max')['c_lift']
        m_features = np.array(m_features.sort_values(0,ascending=False).index)[:top]
        return b.loc[b['variable'].isin(m_features)].reset_index(drop=True), m_features

    def plot_sum(self, min_lift=1, n_var=10, top=5, figsize=(12,4.5), use_cmap=True, fname=None):
        
        '''
        Parameters
        ----------
        
        min_lift : float, optional, default: 1
        \t minimum per-decile lift threshold.
        
        n_var : int, optional, default: 10
        \t maximum number of combined variables
        
        top : int, optional, default: 5
        \t ranking within predefined n ranks
        
        figsize : (float, float), optional, default: (12,4.5)
        \t width, height in inches of plot
        
        use_cmap : boolean, optional, default: True
        \t if True, plot will use predefined color map from matplotlib.
        \t However, if False, plot will use defualt palette.
        
        fname : str or PathLike or file-like object
        \t A path, or a Python file-like object (see pyplot.savefig)
        '''
        gridsize, axis = (1,3), np.full(3,None)
        fig = plt.figure(figsize=figsize)
        axis[0] = plt.subplot2grid(gridsize,(0,0))
        axis[1] = plt.subplot2grid(gridsize,(0,1))
        axis[2] = plt.subplot2grid(gridsize,(0,2))
        var = ['c_lift','pct_target','pct_sample']
        title = [r'$\bf{Lift}$ $\bf{Chart}$',r'$\bf{Target}$',r'$\bf{Sample}$']
        y_label = [r'$\bf{Lift}$',r'$\bf{Cumulative}$ $\bf{Percentage}$ (%)']
        
        a, m_var = self.__filter_vars(min_lift, n_var, top)
        args = [(axis[0], a[['variable',var[0]]], y_label[0], title[0]),
                (axis[1], a[['variable',var[1]]], y_label[1], title[1]),
                (axis[2], a[['variable',var[2]]], y_label[1], title[2])]
        for n in range(3):
            self.__plot_chart(*(args[n]+(n_var, m_var, use_cmap)))
        fig.tight_layout()
        if fname != None: plt.savefig(fname)
        plt.show()
    
    def __plot_chart(self, axis, a, ylabel, title, n_var, m_var, use_cmap):
        
        cmp = self.__cmap(len(m_var))
        kwargs = self.marker.copy()
        for (n,var) in enumerate(m_var,1):
            y = a.loc[a['variable']==var].values[:,1]
            kwargs['label'] = '(%d) %s' % (n,var)
            if use_cmap: kwargs['color'] = cmp[n-1]
            axis.plot(range(len(y)), y, **kwargs)
        axis.set_facecolor('white')
        axis.set_ylabel(ylabel, fontsize=10)
        axis.set_xlabel(r'$\bf{Number}$ $\bf{of}$ $\bf{Variables}$', fontsize=10)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(range(n_var))
        axis.set_xticklabels(np.arange(1,n_var+1), fontsize=10)
        axis.legend(**self.legend2)
        axis.grid(False)
        
    def __cmap(self, n_color=10):
        
        cmap = cm.get_cmap(self.cmap)(np.linspace(0,1,n_color))
        cmap = (cmap*255).astype(int)[:,:3]
        cmap =['#%02x%02x%02x' % (cmap[i,0],cmap[i,1],cmap[i,2]) for i in range(n_color)]
        return cmap
    
def seq_lift(X, y, c_features=None, min_clift=1, min_dlift=0):
    
    '''
    Parameters
    ----------
    
    X : array-like shape of (n_samples , n_features)
    \t binary variables
        
    y : array-like of shape (n_samples,)
    \t target values (binary)
        
    c_features : list of str, optional, default: None
    \t list of starting features. If None, the first variable 
    \t that has the highest per-decile lift will be seleceted.
    
    min_clift : float, optional, default: 1
    \t the minimum cumulative lift required to continue.
    
    min_dlift : float, optional, default: 0
    \t the minimum per-decile lift required to continue.
    
    Returns
    -------
    
    output = tuple of following results
    c_features : list of features 
    cum_target : list of cumulative targets
    cum_sample : list of cumulative samples
    dec_lift : list of per-decile lifts
    pct_target : list of cumulative targets in percentage
    pct_samples : list of cumulative samples in percentage
    cum_lift : list of cumulative lifts
    Note: all lists are arraged according to c_features
    '''
    features = list(X) # all features
    n_target, n_sample = sum(y), len(y)
    target = np.array(y).reshape(-1,1).copy()
    cum_lift, dec_lift = list(), list()
    cum_target, cum_sample = list(), list() 
    if c_features is None: c_features = []
    else:
        a = cumulative_lift(X.copy(), target, c_features)
        cum_target.append(a[0]); cum_sample.append(a[1])
        cum_lift.append(a[2]); dec_lift.append(a[2])
        if len(c_features)>1: c_features = ['_'.join(c_features)]

    while len(c_features)<=len(features):
        
        # remaining features (sorted automatically)
        r_features = list(set(features).difference(c_features))
        # existing flag(s) (if c_feaures == [] then it returns array of zeros)
        existing_flags = (np.sum(X[c_features], axis=1)>0).values.reshape(-1,1)
        # combine existing with new flags 
        combined_flags = ((X[r_features].values + existing_flags)>0) 
        # cumulative number of targets
        c_target = ((combined_flags+target)==2).sum(axis=0)
        # cumulative number of samples
        c_sample = (combined_flags==1).sum(axis=0)
        # cumulative number of existing targets and samples
        ex_target = ((existing_flags + target)==2).sum()
        ex_sample = sum(existing_flags)
        # calculate change of targets and samples
        chg_t = (c_target-ex_target)/n_target 
        chg_s = (c_sample-ex_sample)/n_sample 
        # eliminate sample change that equals to 0
        non_zero = (chg_s>0)
        r_features = np.array(r_features)[non_zero]
        # (1) cumulative lifts
        c_lift = ((c_target/n_target)/(c_sample/n_sample))[non_zero]
        # (2) per-decile lift
        d_lift = chg_t[non_zero]/chg_s[non_zero]
        
        if len(d_lift)>0:
            n = np.argmax(d_lift)
            a = float(c_lift[n]); b = float(d_lift[n])
            if (a>=min_clift) & (b>=min_dlift):
                cum_lift.append(a); dec_lift.append(b)
                cum_target.append(float(c_target[non_zero][n]))
                cum_sample.append(float(c_sample[non_zero][n]))
                c_features.append(r_features[n])
            else: break
        else: break
            
    pct_target = np.array(cum_target)/n_target*100
    pct_sample = np.array(cum_sample)/n_sample*100
    
    return c_features, cum_target, cum_sample, dec_lift, pct_target, pct_sample, cum_lift 

def cumulative_lift(X, y, features):
    
    '''
    Parameters
    ----------
    
    X : array-like shape of (n_samples , n_features)
    \t binary variables
        
    y : array-like of shape (n_samples,)
    \t target values (binary)
        
    features : list of str, optional, default: None
    \t list of features
    
    Return
    ------
    
    output = tuple of (c_target, c_sample, c_lift)
    c_target : cumulative number of targets
    c_sample : cumulative number of samples
    c_lift : cumulative lift
    '''
    target = np.array(y).reshape(-1,1).copy()
    # combine flags within features
    c_flags = (np.sum(X[features],axis=1)>0).values.reshape(-1,1)
    # cumulative number of targets
    c_target = np.sum((c_flags+target)==2)
    # cumulative number of samples
    c_sample = np.sum(c_flags==1)
    # cumulative lift
    c_lift = (c_target/sum(y))/(c_sample/len(y))
    
    return c_target, c_sample, c_lift
