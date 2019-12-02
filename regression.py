import pandas as pd, numpy as np, math, time
import matplotlib.pylab as plt
from scipy.stats import spearmanr, pearsonr
from IPython.display import HTML, display
import ipywidgets as widgets
import statsmodels.discrete.discrete_model as sm
from sklearn.metrics import accuracy_score

#@markdown ##### **_class_** : k_fold

class k_fold:
    
    '''
    Method
    ------

    \t self.fit(X, y): Fit the model according to the given initial inputs.
    \t **Return**
    \t - self.accuracy
    \t - self.model

    K-Folds cross-validator
    -----------------------
    
    sklearn.model_selection.KFold (scikit-learn v0.21.3)
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    '''
    def __init__(self, model, cls_metrics=accuracy_score, 
                 n_splits=10, random_state=0):

      '''
      Parameters
      ----------

      \t model : Regression Model
      \t cls_metrics: classifation metrics (accuracy_score is set as default)
      \t - e.g. metrics.accuracy_score(y_true, y_pred)
      \t n_splits : int, default=10, Number of folds. Must be at least 2.
      \t random_state : int, default=0 RandomState instance or None
      '''
      from sklearn.model_selection import KFold

      self.model, self.cls_metrics = model, cls_metrics
      self.kf = KFold(n_splits=n_splits, shuffle=False, random_state=random_state)
    
    def fit(self, X, y):
      
      '''
      Fitting model
      Parameters
      ----------

      \t X : array-like or sparse matrix, shape (n_samples, n_features)
      \t y : array-like, shape (n_samples)
      '''
      
      self.best_model, self.accuracy, n_kf = None, -1, 0
      for train_index, test_index in self.kf.split(X):
        
        # select sample given indices
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        # fit model
        n_kf += 1
        print(' -- Cross Validation (%d) -- ' % n_kf)
        self.model.fit(X_train, y_train)
        y_proba = self.model.predict_proba(X_test)
        y_proba = y_proba.round()[:,1]
    
        # cut-off @ 50%
        m_accuracy = self.cls_metrics(y_test.values, y_proba)
        print('Model accuracy : %0.2f%%' % (m_accuracy*100))
        if self.accuracy < m_accuracy:
          self.accuracy = m_accuracy
          self.best_model = self.model
        
#@markdown ##### **_class_** : stepwise_logistics

class stepwise_logistics:
  
  '''
  Method
  ------

  \t (1) self.fit(X, y): Fit the model according to the given initial inputs.
  \t **Return**
  \t - self.pvalues : (dataframe)
  \t - self.coef : (dataframe), coefficient of selected variables
  \t - self.intercept : (float), intercept from logistic regression

  \t (2) self.predict_prob(X)
  \t **Return**
  \t - array of probabilities

  Regression model
  ----------------

  Logistic Model (statsmodels v0.11.0dev0)
  https://www.statsmodels.org/dev/_modules/statsmodels/discrete/discrete_model.html#Logit
  '''

  def __init__(self, method='basinhopping', selection='forward', threshold=0.025, fit_intercept=False):
    
    '''
    Parameters
    ----------

    \t method : (str)
    \t - 'newton' : Newton-Raphson
    \t - 'nm' : Nelder-Mead
    \t - 'bfgs' : Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    \t - 'lbfgs' : limited-memory BFGS with optional box constraints
    \t - 'powell' : modified Powell’s method
    \t - 'cg' : conjugate gradient
    \t - 'ncg' : Newton-conjugate gradient
    \t - 'basinhopping' : global basin-hopping solver  (default)
    \t - 'minimize' : generic wrapper of scipy minimize (BFGS by default)

    \t selection : (str)
    \t - 'forward' : Forward (Step-Up) selection (default)
    \t - 'backward' : Backward (Step-Down) selection
    \t - 'stepwise' : a combination of the Forward and Backward selection techniques

    \t threshold : (float), two-tailed hypothesis test alpha
    \t fit_intercept : (bool), whether to calculate the intercept
    '''
    
    self.method, self.selection = method, selection
    self.threshold, self.fit_intercept = threshold, fit_intercept
    self.kwargs = {'disp':False, 'method':method}
    self.inc = 'Attempt (%d) ==> Include (variable : %s , p-value = %.2f%%)'
    self.exc = 'Attempt (%d) ==> Exclude (variable : %s , p-value = %.2f%%)'
  
  def fit(self, X, y):
    
    '''
    Fitting model
    Parameters
    ----------

    \t X : array-like or sparse matrix, shape (n_samples, n_features)
    \t y : array-like, shape (n_samples)
    '''
    # Initiate progress bar
    self.__widgets()
    w = widgets.VBox([self.w_t])
    display(w); time.sleep(5)
    
    if self.selection == 'forward': self.__forward__(y, X)
    elif self.selection == 'backward': self.__backward__(y, X)
    elif self.selection == 'stepwise': self.__stepwise__(y, X)

  def __to_df(self, X):
    
    '''
    Convert Xs to dataframe
    '''
    if isinstance(X, pd.DataFrame)==False: 
      columns = ['X' + str(n+1) for n in range(X.shape[1])]
      X = pd.DataFrame(data=X,columns=columns)
    return X, np.array(X.columns).tolist()
  
  def __pvalues(self, y, X):

    '''
    Determine p-values
    '''
    for n, var in enumerate(self.exclude):
      model = sm.Logit(y, X[self.include + [var]])
      result = model.fit(**self.kwargs)
      if n == 0: pvalues = np.array(result.pvalues)
      else: pvalues = np.vstack((pvalues,result.pvalues))
    return np.array(pvalues).reshape(n+1,len(self.include)+1)
  
  def __min_pvalue(self, pvalues):

    '''
    pvalues is an array (no. of exclude) x (no. of include)
    '''
    # determine the smallest p-value of new entries
    min_index = np.argmin(pvalues[:,-1:],axis=0)[0]
    # a list of p-values
    min_pvalues = pvalues[min_index,:].tolist()
    return min_index, min_pvalues
  
  def __eliminate_var(self, columns, min_index, min_pvalues):
    
    # add new variable, whose p-value is the smallest
    self.include = self.include + [self.exclude[min_index]]
    if self.fit_intercept: new_list = ['Intercept']
    else: new_list = list()
      
    for var, pvalue in zip(self.include, min_pvalues):
      if (var != 'Intercept') & (pvalue <= self.threshold):
        new_list.extend([var])
      else: self.__update_widget((self.exc % (99, var, pvalue*100)))
    self.include = new_list
    self.exclude = [var for var in columns if var not in new_list]
    
  def __thetas(self, y, X):

    model = sm.Logit(y, X[self.include])
    result = model.fit(**self.kwargs)
    self.pvalues = result.pvalues
    self.theta = result.params
    self.mle_retvals = result.mle_retvals
    a = result.params.reset_index().values
    a = pd.DataFrame(a, columns=['variable','coef'])
    a = a.loc[(a['variable']!='Intercept')]
    a['coef'] = a['coef'].astype(float)
    self.coef = a
    self.intercept = a.loc[a['variable']=='Intercept','coef'].values
    if len(self.intercept)==0: self.intercept = 0

  def __stepwise__(self, y, X):
        
    X, self.exclude = self.__to_df(X)
    if self.fit_intercept: 
      X['Intercept'], self.include = 1, ['Intercept']
    else: self.include = list()
    
    n_rnd = 0
    while len(self.exclude) > 0:
      # create list of p-values from all variable combinations
      pvalues = self.__pvalues(y, X)
      # find added variable that has min p-value
      min_index, min_pvalues = self.__min_pvalue(pvalues)
      # p-value of new variable <= threshold, otherwise stops 
      if min_pvalues[-1] <= self.threshold: 
        # update
        n_rnd += 1
        var, pct = self.exclude[min_index], min_pvalues[-1]*100
        self.__update_widget((self.inc % (n_rnd, var, pct)))
        self.__eliminate_var(X.columns, min_index, min_pvalues)
      else: break
    self.__thetas(y, X)
    self.__update_widget('Complete')
    
  def __backward__(self, y, X):
        
    X, self.include = self.__to_df(X)
    if self.fit_intercept: 
      self.include = ['Intercept'] + self.include
      X['Intercept'], n_limit = 1, 1
    else: n_limit = 0
    
    n_rnd = 0
    while len(self.include) > n_limit:
      model = sm.Logit(y, X[self.include])
      result = model.fit(**self.kwargs)
      self.exclude, p_value = self.__find_var(result.pvalues, False)
      if self.exclude != None: 
        # update
        n_rnd += 1
        var, pct = self.exclude, p_value*100
        self.__update_widget((self.exc % (n_rnd, var, pct)))
        # remove variable from the list
        self.include.remove(self.exclude)
      else: break
    self.__thetas(y, X)
    self.__update_widget('Complete')
    
  def __forward__(self, y, X):
        
    X, self.exclude = self.__to_df(X)
    if self.fit_intercept: 
      X['Intercept'], self.include = 1, ['Intercept']
    else: self.include = list()
    
    n_rnd = 0
    while len(self.exclude) > 0:
      # create list of p-values from all variable combinations
      pvalues = self.__pvalues(y, X)
      # find added variable that has min p-value
      min_index, min_pvalues = self.__min_pvalue(pvalues)
      
      # update
      n_rnd += 1
      var, pct = self.exclude[min_index], min_pvalues[-1]*100
      self.__update_widget((self.inc % (n_rnd, var, pct)))
      
      # p-value of new variable <= threshold, otherwise stops
      if min_pvalues[-1] <= self.threshold:
        var = self.exclude[min_index]
        self.include = self.include + [var]
        self.exclude.remove(var)
      else: break
    self.__thetas(y, X)
    self.__update_widget('Complete')
    
  def __find_var(self, pvalues, find_min=True):

    var_name, p_value = None, None
    df = pd.DataFrame(pvalues).reset_index()
    df = df.rename(index=str, columns={'index': 'variable', 0: 'p_value'})
    if find_min == True: cond = (df.p_value <= self.threshold)
    else: cond = (df.p_value > self.threshold)
    df = df.sort_values(by=['p_value','variable'], ascending=find_min)
    df = df.loc[cond & (df.variable != 'Intercept'),['variable','p_value']].values
    if len(df) > 0: 
      var_name, p_value = str(df[0][0]), float(df[0][1])
    return var_name, p_value
    
  def __sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def predict_proba(self, X):

    X, _ = self.__to_df(X)
    if self.fit_intercept == True: X['Intercept'] = 1
    columns = np.array(self.theta.reset_index()['index']).tolist()  
    event = self.__sigmoid(np.dot(X[columns].values, self.theta))
    nonevent = np.array([1-n for n in event]).reshape(-1,1)
    event = np.array(event).reshape(-1,1)
    return np.concatenate((nonevent, event), axis=1)
  
  def __widgets(self):
    self.w_t = widgets.HTMLMath(value='Calculating...')
    
  def __update_widget(self, label):
    self.w_t.value = label; time.sleep(0.1)
    
#@markdown ##### **_class_** : evaluate_classifier

class evaluate_classifier:

  '''
  Method
  ------

  \t self.fit(n_class=1)
  \t - Fit the model according to the given initial inputs and predefinced class
  \t **Return**
  \t - plots of statistical measurement of goodness-to-fit
  '''

  def __init__(self, lb_event='event', lb_nonevent='non-event', width=3.75, height=3.5, n_step=20, 
               c_line ='#ea2027', c_fill ='#7f8fa6'):
    
    '''
    Parameters
    ----------

    \t lb_event: str, target label (default='event')
    \t lb_nonevent: str, non-target label (default='non-event')
    \t width : float, width of plot
    \t height : float, height of plot
    \t n_step : (float) number of bins for distribution plot
    '''
    self.figsize  = (4*width, 2*height)
    self.lb_event, self.lb_nonevent = lb_event, lb_nonevent
    self.n_step = n_step
    self.c_line, self.c_fill = c_line, c_fill
    self.lb_kwargs = dict(loc='best', fontsize=10, framealpha=0, edgecolor='none')

  def fit(self, y, y_proba, n_class=1, fname=None):

    '''
    Parameters
    ----------

    \t y : array-like or list, target array (binary)
    \t y_proba : y : array-like or list, shape (n_samples, {0,1})
    \t n_class : int, class label={0,1}
    \t fname : str or PathLike or file-like object (see pyplot.savefig)
    '''
    self.y, self.y_proba = y, y_proba
    self.n_class = n_class
    print('class (%d) is a target (event)' % self.n_class)
    # find max probability among classes (by row)
    y_pred = np.argmax(self.y_proba, axis=1)
    # prediction given class
    self.y_pred = (y_pred==self.n_class).astype(int)
    # actual target given class
    self.y_true = (self.y==self.n_class).astype(int)
    # probability given class
    self.n_y_proba = self.y_proba[:,self.n_class]
  
    # set up plot area
    fig = plt.figure(figsize=self.figsize)
    axis, b = np.full(8,None), -1
    axis_loc = [(m, n) for m in range(2) for n in range(4)]
    for n, loc in enumerate(axis_loc):
      axis[n] = plt.subplot2grid((2,4),loc)
   
    # (1) ROC curve and GINI
    self._roc_curve(axis[0])

    # (2) Distribution of events and non-events
    self._distribution(axis[1])
    
    # (3) Kolmogorov–Smirnov test
    self._ks(axis[2])

    # (4) Confusion matrix given KS cut-off
    self._confusion_matrix(axis[3], self.ks_cutoff)
    
    # (5) Gain chart 10th to 100th decile
    cum_event, cum_pop, pct_event, pct_pop = self.__cumulative()
    self._gain(axis[4], cum_event, cum_pop)

    # (6) Lift chart (cummulative population)
    ch_lb = 'Lift Chart ($10^{th}$ to $100^{th}$ decile)'
    self._lift(axis[5], cum_event, cum_pop, ch_lb, -1)

    # (7) Lift chart by decile
    ch_lb = 'Lift Chart @ decile'           
    self._lift(axis[6], pct_event, pct_pop, ch_lb, -1, True)  
     
    # (8) 1st to 10th decile
    cum_event, cum_pop, _, _ = self.__cumulative(r_min=90)
    ch_lb = 'Lift Chart ($1^{st}$ to $10^{th}$ decile)'
    self._lift(axis[7], cum_event, cum_pop, ch_lb)
    
    fig.tight_layout()
    if fname is not None: plt.savefig(fname)
    plt.show()
    self.__summary()

  def _confusion_matrix(self, axis, threshold=0.5):

    from sklearn.metrics import confusion_matrix as cfm
    
    r_prob = self.y_proba[:, self.n_class]
    y_pred = (r_prob>=threshold).astype(int)
    
    confusion = cfm(self.y_true, y_pred)
    tp, fp = confusion[1,1], confusion[0,1]
    tn, fn = confusion[0,0], confusion[1,0]
    tt = np.sum(confusion)
    
    self.tp = tp/tt; self.fp = fp/tt
    self.tn = tn/tt; self.fn = fn/tt
    
    # Classification Accuracy: Overall
    # how often is the classifier correct?
    self.accuracy = 100*(tp+tn)/tt 
    # Classification Error: Overall, 
    # how often is the classifier incorrect?
    self.error = 100*(fp+fn)/tt
   
    # Specificity: When the actual value is negative, 
    # how often is the prediction correct?
    self.specificity = 100*tn/max(tn+fp,1)

    # Sensitivity: When the actual value is positive, 
    # how often is the prediction correct?
    # How "sensitive" is the classifier to detecting positive instances?
    self.tpr = 100*tp/max(tp+fn,1)

    # False Positive Rate: When the actual value is negative, 
    # how often is the prediction incorrect?
    self.fpr = 100*fp/max(tn+fp,1)

    # Precision: When a positive value is predicted, 
    # how often is the prediction correct?
    self.precision = 100*tp/max(tp+fp,1)

    axis.matshow(confusion, cmap=plt.cm.hot, alpha=0.2)
    kwargs = dict(va='center', ha='center', fontsize=10)
    label = 'True Positive \n %d (%d%%)' % (tp, 100*self.tp)
    axis.text(0,0, s=label, **kwargs)
    label = 'False Positive \n %d (%d%%)' % (fp,100*self.fp)
    axis.text(1,0, s=label, **kwargs)
    label = 'False Negative \n %d (%d%%)' % (fn,100*self.fn)
    axis.text(0,1, s=label, **kwargs)
    label = 'True Negative \n %d (%d%%)' % (tn,100*self.tn)
    axis.text(1,1, s=label, **kwargs)
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    axis.set_xticks([0.5])
    axis.set_yticks([0.5])
    axis.set_title('$cutoff_{ks}$ = %0.1f%%' % (threshold*100)) 
    axis.set_xlabel('Predicted label')
    axis.set_ylabel('Actual label')
    axis.set_facecolor('white')
    axis.grid(True, lw=1, ls='--', color='k')

  def _precision_recall(self, axis):

    from sklearn.metrics import average_precision_score as p_scr
    from sklearn.metrics import precision_recall_curve as pr_curve

    # compute precision-recall pairs for different probability thresholds
    avg_precision = p_scr(self.y_true, self.y_pred)

    # Precisions and recalls given range of thresholds
    precision, recall, _ = pr_curve(self.y_true, self.n_y_proba)

    # plot results
    axis.step(recall, precision, color=self.c_line, lw=2)
    kwargs = ({'step': 'post'})
    axis.fill_between(recall, precision, color=self.c_fill, alpha=0.2,**kwargs)
    axis.set_xlabel('Recall (Sensitivity)')
    axis.set_ylabel('Precison')
    axis.set_ylim(0,1.1) 
    axis.set_title('Precision-recall curves (%d%%)' % (avg_precision*100))
    axis.set_facecolor('white')
    axis.grid(False)

  def _roc_curve(self, axis):

    from sklearn.metrics import roc_curve, roc_auc_score 

    # ROC curve
    fpr, tpr, roc_thr = roc_curve(self.y_true, self.n_y_proba)
    # Compute Area Under the Receiver Operating Characteristic Curve 
    self.roc_auc = roc_auc_score(self.y_true, self.n_y_proba)
    self.gini = 2*self.roc_auc - 1

    # plot results
    axis.plot(fpr, tpr, color=self.c_line, lw=2, label='ROC curve')
    axis.plot([0,1],[0,1], color=self.c_line, lw=1, 
              label='random classifier', linestyle='--')
    kwargs = dict(step='pre',alpha=0.2,hatch='////',edgecolor='#6b6b6b')
    axis.fill_between(fpr, tpr, color=self.c_fill, **kwargs)
    axis.legend(**self.lb_kwargs)
    axis.set_title('ROC curve (GINI = %d%%)' % (self.gini*100))
    axis.set_xlabel('False Positive Rate (1-Specificity)')
    axis.set_ylabel('True Positive Rate (Sensitivity)')
    axis.set_facecolor('white') 
    axis.grid(False)

  def _distribution(self, axis):

    # distribution of the data
    y_accept = self.y_proba[self.y_pred==1][:,self.n_class]
    if len(y_accept)==0: y_accept=np.zeros(2)
    event = self.y_proba[self.y_true==1][:,self.n_class]
    nonevent = self.y_proba[self.y_true==0][:,self.n_class]
    
    # bins, ticks and tick labels
    bins, x_ticks, x_ticklabels = self.__ticks()
    n_pos, _ = np.histogram(event, bins=bins, range=(0,1))
    n_neg, _ = np.histogram(nonevent, bins=bins, range=(0,1))
    n_pos, n_neg = n_pos/sum(n_pos)*100, n_neg/sum(n_neg)*100
    
    # plot results
    xticks = np.arange(len(bins)-1)
    kwargs = dict(edgecolor='#6b6b6b',alpha=0.5)
    axis.bar(xticks, n_pos, color='#0be881', label=self.lb_event, **kwargs)
    axis.bar(xticks, n_neg, color='#ff3f34', label=self.lb_nonevent, **kwargs)
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_ticklabels)
    axis.legend(**self.lb_kwargs)
    axis.set_title('Distribution ($min_{%s}$ = %d%%)' % (self.lb_event,min(event)*100))
    axis.set_xlabel('Probability')
    axis.set_ylabel('Percentage of samples (%)')
    axis.set_facecolor('white')
    axis.grid(False)
  
  def _ks(self, axis):
    
    # event, non-event and unique bins
    event = self.y_proba[self.y_true==1][:,self.n_class]*100
    nonevent = self.y_proba[self.y_true==0][:,self.n_class]*100
    bins = [0] + np.unique(self.y_proba*100).tolist() + [100]
  
    # cumulative % distribution
    n_pos, _ = np.histogram(event, bins=bins)
    n_neg, _ = np.histogram(nonevent, bins=bins)
    n_pos = [0] + np.cumsum(n_pos/sum(n_pos)).tolist() 
    n_neg = [0] + np.cumsum(n_neg/sum(n_neg)).tolist()
    diff = [abs(m-n) for (m,n) in zip(n_pos,n_neg)]
    n = np.argmax(diff)
    self.ks_cutoff = bins[n]/100
    self.ks = diff[n]
   
    # plot results
    axis.plot(bins, n_pos, color='#0be881', lw=2, label=self.lb_event)
    axis.plot(bins, n_neg, color='#ff3f34', lw=2, label=self.lb_nonevent)
    kwargs = dict(alpha=0.2, hatch='////', edgecolor='#6b6b6b')
    axis.fill_between(bins, n_pos, n_neg,color=self.c_fill,**kwargs)
    axis.plot(bins, diff, color='#3742fa', lw=1, label='KS')
    axis.axvline(bins[n], color='k',linestyle="--", lw=0.8) 
    axis.legend(**self.lb_kwargs)
    value_tp = tuple((diff[n]*100, self.ks_cutoff*100))
    axis.set_title('KS=%0.2f%%, cutoff=%0.2f%%' % value_tp)
    axis.set_xlabel('Probability')
    axis.set_ylabel('Cumulative Distribution (%)')
    axis.set_facecolor('white')
    axis.grid(False)
  
  def __ticks(self, n_ticks=5):
    
    # create bins given number of steps
    r_prob = self.y_proba[:, self.n_class]
    r_max, r_min = max(r_prob), min(r_prob)
    r_incr = (r_max-r_min)/self.n_step
    bins = [r_min + (n*r_incr) for n in range(self.n_step+1)] 
    bins[-1] = bins[-1] * 1.01
    
    # tick positions and labels
    xincr = (r_max-r_min)/n_ticks
    xticks = [n*(len(bins)-1)/n_ticks for n in range(n_ticks+1)]
    xticklabels = [r_min + (n*xincr) for n in range(n_ticks+1)]
    xticklabels = ['{:,.0f}'.format(n*100) for n in xticklabels] 
    
    return bins, xticks, xticklabels
  
  def _gain(self, axis, cum_event, cum_pop):
    
    # multiply with 100 (convert % to integer)
    cum_event = [0] + [int(n*100) for n in cum_event]
    cum_pop = [0] + [round(int(n*100),-1) for n in cum_pop]
    
    # plot results
    xticks = np.arange(len(cum_event))
    kwargs = dict(lw=2, label='model', marker='o',markersize=5)
    axis.plot(xticks, cum_event, color=self.c_line, **kwargs)
    axis.plot([0,10],[0,100], color=self.c_line, lw=1, label='random', ls='--')
    kwargs = dict(alpha=0.2,hatch='////',edgecolor='#6b6b6b')
    axis.fill_between(xticks, cum_event, color=self.c_fill, **kwargs)
    axis.set_xticks(xticks)
    axis.set_xticklabels(cum_pop)
    axis.legend(**self.lb_kwargs)
    axis.set_title('Gain Chart')
    axis.set_xlabel('% of datasets (decile)')
    axis.set_ylabel('Cumulative % of events')
    axis.set_facecolor('white')
    axis.grid(False)

  def _lift(self, axis, event, pop, label='Lift Chart', digit=0, cum=False):
    
    # caluclate lift
    lift = [m/n for (m,n) in zip(event,pop)]
    if cum==True: pop = np.cumsum(pop)
    cum_pop = [round(int(n*100), digit) for n in pop]
    
    # plot results
    xticks = np.arange(len(lift))
    axis.plot(xticks, lift, color=self.c_line, lw=2, 
              label='model', marker='o',markersize=5)
    axis.plot([0,9],[1,1], color=self.c_line, lw=1, label='random', ls='--')
    kwargs = dict(hatch='////', edgecolor='#6b6b6b', alpha=0.2)
    axis.fill_between(xticks, lift, color=self.c_fill, **kwargs)
    axis.set_xticks(xticks)
    axis.set_xticklabels(cum_pop)
    axis.legend(**self.lb_kwargs)
    axis.set_title(label)
    axis.set_xlabel('% of datasets (decile)')
    axis.set_ylabel('Lift')
    axis.set_facecolor('white')
    axis.grid(False)
  
  def __cumulative(self, r_min=0, r_max=100):
    
    ''' 
    Gain at a given decile level is the ratio of cumulative 
    number of targets (events) up to that decile to the total 
    number of targets (events) in the entire data set
    
    Lift measures how much better one can expect to do with 
    the predictive model comparing without a model. It is the ratio 
    of gain % to the random expectation % at a given decile level. 
    The random expectation at the xth decile is x%
    '''
    r_prob = self.y_proba[:, self.n_class]
    r_incr = (r_max-r_min)/10
    bin_pct = [(r_min + r_incr*n) for n in range(11)]
    bins = [np.percentile(r_prob, n) for n in bin_pct]
    event = self.y_proba[self.y_true==self.n_class][:,self.n_class]
    n_event, _ = np.histogram(event, bins=bins)
    n_pop, _ = np.histogram(r_prob, bins=bins)
    
    # Cumulative number of events and populations
    pct_event = n_event[::-1]/len(event)
    pct_pop = n_pop[::-1]/len(r_prob)
    cum_event = np.cumsum(pct_event).tolist()
    cum_pop = np.cumsum(pct_pop).tolist()
    
    return cum_event, cum_pop, pct_event, pct_pop
  
  def __summary(self):
    
    columns = ['TP','FP','TN','FN','accuracy','error','specificity',
               'tpr','fpr','precision', 'roc', 'gini', 'ks', 'ks_cutoff']
    a = np.array([self.tp, self.fp, self.tn, self.fn, self.accuracy, 
                  self.error, self.specificity, self.tpr, self.fpr, 
                  self.precision, self.roc_auc, self.gini, self.ks, 
                  self.ks_cutoff]).reshape(1,-1)
    self.df = pd.DataFrame(a,columns=columns)
