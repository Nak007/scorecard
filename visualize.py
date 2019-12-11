import pandas as pd, numpy as np
import matplotlib.pylab as plt

class target_rate:
    
    '''
    Method
    ------
    
    \t self.fit(X, y, fname=None)
    \t **Return**
    \t - plot histogram as well as target rate
    '''
    def __init__(self, bins=10, width=4, height=3.5, axis_cols=4, 
                 show_pct=False, show_legend=True):
        
        '''
        Parameters
        ----------
        
        \t bins : (int), maximum number of bins (default=10)
        \t width, height : (float), width and height of plot (default=(4,3.5))
        \t axis_cols : (int), number of columns when plots are displayed (default=4)
        \t show_pct : (bool), percentage of histogram is visible when True (default=False)
        \t show_legend : (bool), legend is visible when True (default=True)
        '''
        self.bins = bins
        self.width, self.height = width, height
        self.axis_cols = axis_cols
        self.show_pct, self.show_legend = show_pct, show_legend
        b_kwargs = dict(width=0.5, hatch='/////', alpha=0.6, lw=1, edgecolor='#708090')
        self.n_kwargs = b_kwargs.copy(); self.n_kwargs.update(dict(color='#00ff9d', label='w/o nan'))
        self.t_kwargs = b_kwargs.copy(); self.t_kwargs.update(dict(color='#ff009d', label='nan'))
        self.a_kwargs = dict(color='k', ha='center',va='bottom', fontsize=11)
        self.l_kwargs = dict(color='k', lw=1, marker='o', markersize=5, ls='--', label='%target', markerfacecolor='none')
        self.lg_kwargs = dict(loc='best', fontsize=10, framealpha=0, edgecolor='none')
    
    def fit(self, X, y, fname=None):
        
        '''
        plot histogram as well as target rate
        
        Parameters
        ----------
        
        \t X : (dataframe) of shape (n_samples, n_features)
        \t y : (series), Target values
        \t fname : str or PathLike or file-like object (see pyplot.savefig)
        '''
        a = self.__numeric_var(X.copy())
        axis_row = int(np.ceil(a.shape[1]/self.axis_cols))
        loc_list = [(n,m) for n in range(axis_row) for m in range(self.axis_cols)]
        figsize = (self.axis_cols*self.width,axis_row*self.height)
        fig = plt.figure(figsize=figsize)
        
        for n,loc in enumerate(loc_list[:a.shape[1]]):
            axis = plt.subplot2grid((axis_row,self.axis_cols),loc)
            self.__plot_rate(axis, y, a[list(a)[n]])
            
        plt.tight_layout()
        if fname!=None: plt.savefig(fname)
        plt.show()
    
    def __plot_rate(self, axis, y, x):

        # y and x with nan and no-nan
        nonan, nan = ~np.isnan(x), np.isnan(x)
        nonan_y, nonan_x = y[nonan], x[nonan]
        nan_y, nan_x, n_sample = y[nan], x[nan], len(y)

        # determine number of bins
        unq_bins = np.unique(nonan_x)
        bins = min(max(2,len(unq_bins)),self.bins)
        a, bin_edges = np.histogram(nonan_x,bins=bins)

        # x ticks and labels
        x_ticks = range(len(a)+1)
        x_labels = ['nan'] + ['{:,.2f}'.format(n) if n < 999 else 
                              '{:,.0e}'.format(n) for n in bin_edges[:-1]]

        # plot histogram (nan)
        pct_nonan = np.array([0] + list(a))/n_sample*100
        bar1 = axis.bar(x_ticks, pct_nonan, **self.n_kwargs)
        pct_nan = [len(nan_y)/n_sample*100] + np.full(len(a),0).tolist()
        bar2 = axis.bar(x_ticks, pct_nan,**self.t_kwargs)
        
        # display percentage above histogram
        if self.show_pct:
            for (nx,ny) in zip(x_ticks[1:],pct_nonan[1:]):
                if ny>0: axis.annotate('%0.1f' % ny,(nx,ny), **self.a_kwargs)
 
        axis.set_facecolor('white')
        axis.set_ylabel('Percentage (%)', fontsize=10)
        axis.set_xlabel(r'$BIN_{n}$ $\leq$ X < $BIN_{n+1}$')
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_labels, fontsize=10, rotation=90)
        axis.grid(False)

        # plot target rate
        tw_axis = axis.twinx()
        tw_axis.set_ylabel('Target Rate (%) by BIN', fontsize=10)
        t_group = np.digitize(nonan_x,bins=bin_edges)[nonan_y==1]
        t_nan = (nan_y==1).sum()/max(len(nan_y),1)*100
        target_rate = [(t_group==n).sum()/max(d,1)*100 for n,d in zip(x_ticks,a)]
        line1 = tw_axis.plot(x_ticks[1:], target_rate, **self.l_kwargs)
        title = '%s \n nan=%0.2f%%, target=%0.2f%%' % (x.name, len(nan_y)/n_sample*100,t_nan)
        axis.set_title(title, fontsize=11)
        
        # legend
        if self.show_legend:
            line = [bar1] + [bar2] + line1
            labels = [m.get_label() for m in line]
            tw_axis.legend(line, labels, **self.lg_kwargs)

    def __numeric_var(self, a):
        
        columns = list()
        for var in a.columns:
            try: a[var] = a[var].astype(float)
            except: pass
            if a[var].dtype in ['float64','int64']: columns.append(var)
        return a[columns]
