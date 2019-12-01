import pandas as pd, numpy as np, math, time
import scipy.stats as st, ipywidgets as widgets
from IPython.display import HTML, display

#@markdown **_class_** : batch_evaluation

class batch_evaluation:

  '''
  Method
  ------

  \t self.fit( y, X)
  \t **Return:** 
  \t - self.bin_df : (dataframe), table of hyper parameters and goodness-of-fit 
  \t   or binning indicators from each iteration. There are 9 columns, 
  \t   which are 'round', 'method', 'variable', 'min_pct', 'step', 'model_bin', 
  \t   'IV', 'correlation', and 'intercept'
  \t - self.res_df : (dataframe), table of all binning outputs. Table is conprised of 
  \t   'round', 'variable', 'min', 'max', 'Bin', 'Non_events', 'Events', 
  \t   'pct_nonevents', 'pct_events', 'WOE', and 'IV'

  \t self.plot(column='round', value=1, adjusted=False)
  \t **Return**
  \t - self.adj_bin_df : (dataframe)

  \t self.filter_out()
  \t **Return**
  \t - self.adj_bin_df : (dataframe)
  '''
  def __init__(self, n_step=20, method=list(['iv','entropy','gini','chi','mono']),
               min_pct=0.05, chi_alpha=0.05, chi_intv=15, p_value=0.05, ttest_intv=15, 
               min_obs=0.05, min_event=0.05, iv_imp=0.5, min_iv=0.1, min_corr=0.5, max_tol=0.01,
               bin_df=None, res_df=None):
    '''
    Parameters
    ----------

    **woe_binning**
    \t n_step : (int), number of steps (percentile) given defined range (min, max)
    \t method : (str,list) method to use for determining WOEs (default=['iv','entropy','gini','chi','mono'])
    \t - string method name
    \t - list of method names e.g. ['iv', 'gini']
    \t min_pct : (float), minimum percentage of sample in each BIN
    \t chi_alpha : (float), significant level of Chi-sqaure
    \t chi_intv : (int), starting sub-intervals of Chi-merge
    \t p_value : (float), significant level of Student's T-Test
    \t ttest_intv : (int), starting sub-intervals of "Monotonic-Optimal-Binning" technique
    \t min_obs : (float), minimum percentage of sample in each BIN (used in 'mono')
    \t min_event : (float), minimum percentage of event compared to its total in each BIN

    **Selection criteria**
    \t iv_imp : (float), IV importance weight (max=1, default=0.5)
    \t min_iv : (float), minimum acceptable IV (default=0.1)
    \t min_corr : (float), minimum acceptable absolute correlation (default=0.5)
    \t max_tol : (float), maximum tolerance of difference between model and 
    \t log(event/non-event) intercepts (default=0.01)
    \t bin_df : (dataframe), list of hyper parameters and goodness-of-fit or 
    \t binning indicators from each iteration (default=None)
    \t res_df : (dataframe), list of binning outputs (default=None)
    '''
    # keyword arguments (excluding 'method')
    self.kwargs = dict(n_step=n_step, min_pct=min_pct, chi_alpha=chi_alpha, 
                       chi_intv=chi_intv, p_value=p_value, ttest_intv=ttest_intv, 
                       min_obs=min_obs, min_event=min_event)           

    # Selection criteria
    self.iv_imp = max(iv_imp,1) # iv importance weight
    self.min_iv = min_iv # min. information value
    self.min_corr = min_corr # min. absolute correlation
    self.max_tol = max_tol # max. tolerance for intercept
    
    # External input of dataframes
    self.res_df, self.bin_df = res_df, bin_df
    
    # list of columns 
    self.bin_cols=['round', 'method', 'variable', 'model_bin', 'IV', 'correlation', 'intercept']
    self.res_cols=['round', 'variable', 'min', 'max', 'Bin', 'Non_events',
                   'Events', 'pct_nonevents', 'pct_events', 'WOE', 'IV']
    
    # Set label format
    self.prog_lb = '** Variable: {var} (method={m}) **'

    # define method
    if isinstance(method,str): self.method = list([method])
    elif isinstance(method,list): self.method = method
    else: self.method = ['iv','entropy','gini','chi','mono']

  def fit(self, y, X):
    
    '''
    Parameters
    ----------

    \t y : array-like, shape (n_samples)
    \t X : {array-like, sparse matrix}, shape (n_samples, n_features)
    '''
    # Assign value to variables
    self.__widgets()
    y = np.array(y)
    X, columns = self.__to_df(X)
    t_round, n_round = len(columns)*len(self.method) , 0

    # dataFrames for WOE results and list of hyper-parameters
    self.bin_df = pd.DataFrame(columns=self.bin_cols)
    self.res_df = pd.DataFrame(columns=self.res_cols)

    for n_method in self.method:
      
      for var in columns:
        
        # update progress bar
        n_round += 1
        self.__update_widget(n_round/t_round, self.prog_lb.format(var=var,m=n_method))
        
        # initial values
        self.kwargs['method'] = n_method
        bin_model = woe_binning(**self.kwargs)
      
        # determine coarse binning
        bin_model.fit(y, X[var].values)
        md = evaluate_bins(y, X[var].values, bin_model.bin_edges)
        
        # find difference of intercepts
        try: incpt = abs((md.modl_incpt-md.ln_incpt)/max(float(md.modl_incpt),1))
        except: incpt = np.nan
        model_bin, iv, corr = len(md.woe_df)-1, md.iv, md.rho
        woe_df = md.woe_df.copy()
        
        # high-level results
        new_df = {'round':n_round,'method':n_method,'variable':var,
                  'model_bin':len(md.woe_df)-1,'IV':md.iv,
                  'correlation':md.rho,'intercept':incpt}
        self.bin_df = self.bin_df.append(new_df, ignore_index=True)
        
        # detailed results
        woe_df['round'], woe_df['variable'] = n_round, var
        self.res_df = self.res_df.append(woe_df[self.res_cols], ignore_index=True)
    
    # apply selection criteria
    self.filter_out()

  def filter_out(self):
    
    '''
    Apply selection criteria to determine the best possible outcome
    Criteria are as follow
    (1) exceeds minimum Information Value (IV)
    (2) exceeds minimum absolute correlation
    (3) remain within maximum tolerance for intercept

    Subsequently, importance weight is applied and its condition is expressed as follows
    \t $iv_{n} \geq iv_{min}$ and $|\rho_{n}| \geq |\rho_{min}|$ and $\delta_{intercept} \leq x\%$
    Given the same 'variable' or 'round' that provides the highest score is selected 
    where 
    \t $score_{n} = w_{iv,n}(iv_{n}) + (1-w_{iv,n})|\rho_{n}|$
    \t $n$ : nth round
    \t $w_{iv}$ : importance weight of $iv$ (Information Value)
    \t $|\rho| $ absolute Pearson correlation coefficient

    Last but not least, if there is more than one output (per variable) left as 
    abovementioned, the earliest '_round_' is selected.
    '''
    bin_df = self.bin_df.rename(str.lower,axis=1).copy()
    
    # set out conditions
    cond = (bin_df['iv']>=self.min_iv) 
    cond = cond & (abs(bin_df['correlation'])>=self.min_corr)
    cond = cond & (bin_df['intercept']<=self.max_tol)
    bin_df = bin_df.loc[cond]
    
    # weighted score between correlation and iv
    bin_df['w_scr'] = abs(bin_df['correlation'])*(1-self.iv_imp) 
    bin_df['w_scr'] = bin_df['w_scr'] + bin_df['iv']*self.iv_imp
    
    # select method, whose weighted score is the highest
    a = bin_df[['variable','w_scr']].groupby(['variable']).agg(['max'])
    a = pd.DataFrame(a.reset_index().values, columns=['variable','max_scr'])
    a = bin_df.merge(a, on='variable', how='left')
    a = a.loc[(a['w_scr']==a['max_scr'])].drop(columns=['w_scr','max_scr'])
    
    # if there is more than one candidate left, select the first one
    a = a[['variable','round']].groupby(['variable']).agg(['min'])
    a = a.values.reshape(1,-1).tolist()[0]
    bin_df = bin_df.loc[(bin_df['round'].isin(a))].drop(columns=['w_scr'])
    self.adj_bin_df = bin_df.reset_index(drop=True)

  def plot(self, column='round', value=1, adjusted=False):
    
    '''
    Plot WOE from selected list
    Selection can be made from either 'round' or 'varialbe' column
    Value must correpond to selected columns (str, int, or list-like)

    Parameters
    ----------

    \t column : (str), column in bin_df {'_round_', '_variable_'}
    \t value : (list or int) column = '_round_' or (str) e.g. column = '_variable_'
    \t adjusted : (boolean), if True, adj_bin_df (filtered dataframe) is used instead.
    '''
    if isinstance(value,(str, int)): value = np.array([value])
    if column not in ['round','variable']: column = 'round' 
      
    # select list   
    if adjusted == False: bin_df = self.bin_df
    else: bin_df = self.adj_bin_df
    
    for n in value:
      # select dataset given column and its value
      a = bin_df.loc[(bin_df[column]==n)]
      # count how many rounds there are
      n_round = np.unique(a['round'].values)
      if n_round.size > 0:
        for i, m in enumerate(n_round):
          b = a.loc[(a['round']==m)]
          var_name = str(b['variable'].values)
          rho = float(b['correlation'].values)
          method = str(b['method'].values)
          woe_df = self.res_df.loc[(self.res_df['round']==m)]
          print('method: %s (round=%d)' % (method,m))
          plot_woe().plot(woe_df, var_name, rho=rho) 
      else: print('[Empty list]: %s' % n)
  
  def __widgets(self):
    
    '''
    Initialize widget i.e. progress-bar and text
    '''
    kwargs = dict(value=0, min=0, max=100, step=0.01, 
                  bar_style='info', orientation='horizontal')
    self.w_t = widgets.HTMLMath(value='Calculating...')
    self.w_f = widgets.FloatProgress(**kwargs)
    w = widgets.VBox([self.w_t, self.w_f])
    display(w); time.sleep(5)
  
  def __update_widget(self, pct_val, n_label=None):
    
    '''
    Update widget
    '''
    self.w_f.value = pct_val * 100
    self.w_t.value = '({:.0f}%) '.format(pct_val*100) + n_label 
    time.sleep(0.1)
    if pct_val == 1: 
      self.w_f.bar_style = 'success'
      self.w_t.value = '(100%) Complete'
      
  def __to_df(self, X):
    
    '''
    if X is an array, it will be transformed to dataframe.
    Name(s) will be automatically assigned to all columns i.e. X1, X2, etc.
    '''
    if isinstance(X, pd.core.series.Series):
      X = pd.DataFrame(X)
    elif isinstance(X, pd.DataFrame)==False:
      try: n_col = X.shape[1]
      except: n_col = 1
      columns = ['X' + str(n+1) for n in range(n_col)]
      X = pd.DataFrame(data=np.array(X),columns=columns) 
    return X, np.array(X.columns).tolist()
