import pandas as pd, numpy as np, math, time
import matplotlib.pylab as plt
from scipy.stats import spearmanr, pearsonr
from IPython.display import HTML, display
import ipywidgets as widgets
import statsmodels.discrete.discrete_model as sm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def k_iter_logit(estimator, X, y, cls_metrics=accuracy_score, n_splits=10, random_state=0):
    
    '''
    k_iter_logit is built for making scorecard only. a iteration where 
    positive coefficients are eliminated in each round
    
    Parameters
    ----------
    
    estimator : regression estimator object
    \t regression estimator e.g. sklearn.linear_model.LogisticRegression
        
    X : dataframe, shape of (n_samples, n_features)
        
    y : array-like, shape of (n_samples)
    
    cls_metrics : classification metrics object (sklearn.metrics)
    \t classifation metrics (accuracy_score is set as default)
    \t e.g. metrics.accuracy_score(y_true, y_pred)

    n_splits : int, optional, default: 5 
    \ number of folds. Must be at least 2

    random_state : int, optional ,default: 0 
    \t randomState instance or None
    '''
    variables, pos_coef, n = X.columns, 100, 0
    cv_label = 'Round ==> {:,d}, no. of variables: {:,d}'
    while pos_coef > 0:
        n += 1; print(cv_label.format(n,len(variables)))
        k = k_fold(estimator, cls_metrics, n_splits, random_state)
        k.fit(X[variables].copy(), y.copy())
        pos_coef = np.sum(k.best_model.coef_[0]>0)
        variables = variables[(k.best_model.coef_[0]<=0)]
        
    # result: coefficients and weights
    a = (np.array(variables).reshape(-1,1),
         k.best_model.coef_[0].reshape(-1,1))
    theta = pd.DataFrame(np.hstack(a), columns=['variable','coef'])
    theta['weight'] = abs(theta['coef']/np.sum(abs(theta['coef']))*100)
    theta = theta.sort_values(by=['weight'],ascending=False).reset_index(drop=True)
      
    return k.best_model, variables, theta

class k_fold:

    '''
    Method
    ------

    self.fit(X, y)
    \t fit model according to the given initial inputs

    K-Folds cross-validator
    -----------------------

    sklearn.model_selection.KFold (scikit-learn v0.21.3)
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    '''
    def __init__(self, estimator, cls_metrics=accuracy_score, n_splits=5, random_state=0):

        '''
        Parameters
        ----------

        estimator : estimator object
        \t estimator e.g. sklearn.linear_model.LogisticRegression
        
        cls_metrics : classification metrics object (sklearn.metrics)
        \t classifation metrics (accuracy_score is set as default)
        \t e.g. metrics.accuracy_score(y_true, y_pred)
        
        n_splits : int, optional, default: 5 
        \ number of folds. Must be at least 2
        
        random_state : int, optional ,default: 0 
        \t RandomState instance or None
        '''
        self.model, self.cls_metrics = estimator, cls_metrics
        self.cls_name = self.cls_metrics.__name__
        self.kf = KFold(n_splits=max(n_splits,2), shuffle=False, random_state=random_state)
        self.str = '- Cross Validation (%d) --> Model performance (%s) : %0.2f%%'
    
    def fit(self, X, y):

        '''
        Parameters
        ----------

        X : array-like or sparse matrix, shape of (n_samples, n_features)
        
        y : array-like, shape of (n_samples)
        
        Return
        ------
        
        self.m_perform
        \t best estimator's performace given the classification metrics
        
        self.best_model
        \t best estimator object
        '''
        self.best_model, self.m_perform, n_kf = None, -1, 0
        for train_index, test_index in self.kf.split(X):

            # select sample given indices
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            # fit model
            self.model.fit(X_train, y_train)
            y_proba = self.model.predict_proba(X_test)
            y_proba = y_proba.round()[:,1]

            # cut-off @ 50%
            m_perform = self.cls_metrics(y_test.values, y_proba)
            n_kf += 1; print(self.str % (n_kf, self.cls_name ,m_perform*100))
            if self.m_perform < m_perform:
                self.m_perform = m_perform
                self.best_model = self.model

class stepwise_logistics:

    '''
    Method
    ------

    self.fit(X, y): Fit the model according to the given initial inputs.
    self.predict_prob(X)

    Regression model
    ----------------

    Logistic Model (statsmodels v0.11.0dev0)
    https://www.statsmodels.org/dev/_modules/statsmodels/discrete/discrete_model.html#Logit
    '''

    def __init__(self, method='basinhopping', selection='forward', threshold=0.025, fit_intercept=False):

        '''
        Parameters
        ----------

        method : str, optional, default: 'basinhopping'
        \t 'newton' : Newton-Raphson
        \t 'nm' : Nelder-Mead
        \t 'bfgs' : Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        \t 'lbfgs' : limited-memory BFGS with optional box constraints
        \t 'powell' : modified Powell’s method
        \t 'cg' : conjugate gradient
        \t 'ncg' : Newton-conjugate gradient
        \t 'basinhopping' : global basin-hopping solver
        \t 'minimize' : generic wrapper of scipy minimize (BFGS by default)

        selection : str, optional, default: 'forward'
        \t 'forward' : Forward (Step-Up) selection
        \t 'backward' : Backward (Step-Down) selection
        \t 'stepwise' : a combination of the Forward and Backward selection techniques

        threshold : float, optional, default: 0.025 (2.5%) 
        \t two-tailed hypothesis test alpha
        
        fit_intercept : bool, optional, default: False
        \t whether to calculate the intercept
        '''
        self.method, self.selection = method, selection
        self.threshold, self.fit_intercept = threshold, fit_intercept
        self.kwargs = {'disp':False, 'method':method}
        self.inc = 'Attempt (%d) ==> Include (variable : %s , p-value = %.2f%%)'
        self.exc = 'Attempt (%d) ==> Exclude (variable : %s , p-value = %.2f%%)'

    def fit(self, X, y):

        '''
        Parameters
        ----------

        X : array-like or sparse matrix, shape (n_samples, n_features)
        y : array-like, shape (n_samples)
        
        Return
        ------
        
        self.pvalues : dataframe
        self.coef : dataframe, coefficient of selected variables
        self.intercept : float, intercept from logistic regression
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
            if (var != 'Intercept') & (pvalue <= self.threshold): new_list.extend([var])
            else: self.__update_widget((self.exc % (99, var, pvalue*100)))
        self.include = new_list
        self.exclude = [var for var in columns if var not in new_list]
    
    def __thetas(self, y, X):
        
        # fit model with included variables
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
        if self.fit_intercept: X['Intercept'], self.include = 1, ['Intercept']
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
        if self.fit_intercept: X['Intercept'], self.include = 1, ['Intercept']
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
        if len(df) > 0: var_name, p_value = str(df[0][0]), float(df[0][1])
        return var_name, p_value
    
    def predict_proba(self, X):
        
        '''
        Probability estimates.
        The returned estimates for all classes are ordered by the label of classes.
        
        Parameters
        ----------
        
        X : (dataframe) of shape (n_samples, n_features)
        \t Vector to be scored, where n_samples is the number of samples 
        \t and n_features is the number of features.
        '''
        X, _ = self.__to_df(X)
        if self.fit_intercept == True: X['Intercept'] = 1
        columns = np.array(self.theta.reset_index()['index']).tolist()  
        event = self.__sigmoid(np.dot(X[columns].values, self.theta))
        nonevent = np.array([1-n for n in event]).reshape(-1,1)
        event = np.array(event).reshape(-1,1)
        return np.concatenate((nonevent, event), axis=1)
      
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __widgets(self):
        self.w_t = widgets.HTMLMath(value='Calculating...')

    def __update_widget(self, label):
        self.w_t.value = label; time.sleep(0.1)
        
class evaluate_classifier:

    '''
    Method
    ------

    self.fit(n_class=1)
    \t Fit the model according to the given initial inputs and predefinced class
    **Return**
    \t plots of statistical measurement of goodness-to-fit
    '''

    def __init__(self, lb_event='event', lb_nonevent='non-event', figsize=(3.75,3.5), n_step=20, 
                 c_line ='#ea2027', c_fill ='#7f8fa6'):

        '''
        Parameters
        ----------

        lb_event : str, optional, default: 'event'
        \t target label 
        
        lb_nonevent : str, optional, default: 'non-event'
        \t non-target label
        
        figsize : tuple of floats, optional, default: (3.75,3.5)
        \t width and height of each plot (inches)
        
        n_step : float, optional,  default: 20
        \t number of bins for distribution plot
        
        c_line : hex, optional, default: '#ea2027'
        \t color of line plot
        
        c_fill : hex, optional, default: '#7f8fa6'
        \t color of fill plot 
        '''
        self.figsize  = (4*figsize[0], 2*figsize[1])
        self.lb_event, self.lb_nonevent = lb_event, lb_nonevent
        self.n_step = n_step
        self.c_line, self.c_fill = c_line, c_fill
        self.lb_kwargs = dict(loc='best', fontsize=10, framealpha=0, edgecolor='none')

    def fit(self, y, y_proba, n_class=1, fname=None):

        '''
        Parameters
        ----------

        y : array-like or list, target array (binary)
        
        y_proba : array-like of shape (n_samples, n_classes)
        \t log-probability of the sample for each class in the model
        \t where classes are ordered from 0 to (n-1) class
        
        n_class : int, optional, default: 1
        \t class label
        
        fname : str or PathLike or file-like object, default: None
        \t (see pyplot.savefig)
        '''
        self.y, self.y_proba = np.array(y), np.array(y_proba)
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
        ch_lb = r'Lift Chart ($10^{th}$ to $100^{th}$ decile)'
        self._lift(axis[5], cum_event, cum_pop, ch_lb, -1)

        # (7) Lift chart by decile      
        self._lift(axis[6], pct_event, pct_pop, 'Lift Chart @ decile', -1, True)  

        # (8) 1st to 10th decile
        cum_event, cum_pop, _, _ = self.__cumulative(r_min=90)
        ch_lb = r'Lift Chart ($1^{st}$ to $10^{th}$ decile)'
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
        label = 'True Positive \n %s (%d%%)' % ('{:,.0f}'.format(tp), 100*self.tp)
        axis.text(0,0, s=label, **kwargs)
        label = 'False Positive \n %s (%d%%)' % ('{:,.0f}'.format(fp),100*self.fp)
        axis.text(1,0, s=label, **kwargs)
        label = 'False Negative \n %s (%d%%)' % ('{:,.0f}'.format(fn),100*self.fn)
        axis.text(0,1, s=label, **kwargs)
        label = 'True Negative \n %s (%d%%)' % ('{:,.0f}'.format(tn),100*self.tn)
        axis.text(1,1, s=label, **kwargs)
        axis.set_xticks([0,1])
        axis.set_yticks([0,1])
        axis.set_yticklabels(['True','False'], rotation=90)
        axis.set_xticklabels(['True','False'])
        axis.set_ylabel(r'Predict ($cutoff_{ks}$ = %0.1f%%)' % (threshold*100))
        axis.set_xlabel('Actual')
        axis.set_facecolor('white')

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
        axis.plot([0,1],[0,1], color=self.c_line, lw=1, label='random classifier', linestyle='--')
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
        
        '''
        Gain chart plot the cumulative of number of targets (events) 
        against the cumulative number of samples 
        '''
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
        axis.set_xlabel('Cumulative % of datasets')
        axis.set_ylabel('Cumulative % of events')
        axis.set_facecolor('white')
        axis.grid(False)

    def _lift(self, axis, event, pop, label='Lift Chart', digit=0, cum=False):
        
        ''' 
        Lift measures how much better one can expect to do with 
        the predictive model comparing without a model. It is the ratio 
        of gain % to the random expectation % at a given decile level. 
        The random expectation at the xth decile is x%
        '''
        # caluclate lift
        lift = [m/n for (m,n) in zip(event,pop)]
        if cum==True: pop = np.cumsum(pop)
        cum_pop = [round(int(n*100), digit) for n in pop]

        # plot results
        xticks = np.arange(len(lift))
        axis.plot(xticks, lift, color=self.c_line, lw=2, label='model', marker='o',markersize=5)
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

        y_proba = np.array(self.y_proba[:,self.n_class]).reshape(-1,1)
        y_true = np.array(self.y_true).reshape(-1,1).copy()
        columns = ['y_proba','y_true']
        y = pd.DataFrame(np.hstack((y_proba, y_true)),columns=columns)
        y = y.sort_values(by=columns).reset_index(drop=True)
        event, sample, index = (y['y_true']==1).sum(), len(y), range(len(y))
        bin_pct = [(r_min + (r_max-r_min)/10*n) for n in range(11)]
        bin_index = [int(np.percentile(index,n)) for n in bin_pct]
        n_event, n_sample = list(), list()
        for n,m in zip(bin_index[:-1],bin_index[1:]):
            b = y.iloc[n:m+1,:]
            n_sample.append(b.shape[0])
            n_event.append(b.loc[b['y_true']==1].shape[0])

        # Cumulative number of events and populations
        p_event = np.array(n_event)[::-1]/event
        p_sample = np.array(n_sample)[::-1]/sample
        c_event = np.cumsum(p_event).tolist()
        c_sample = np.cumsum(p_sample).tolist()
        return c_event, c_sample, p_event, p_sample

    def __summary(self):

        a = ['TP','FP','TN','FN','accuracy','error','specificity',
             'tpr','fpr','precision', 'roc', 'gini', 'ks', 'ks_cutoff']
        b = [self.tp, self.fp, self.tn, self.fn, self.accuracy, self.error, 
             self.specificity, self.tpr, self.fpr, self.precision, self.roc_auc, 
             self.gini, self.ks, self.ks_cutoff]
        self.df = pd.DataFrame({'measurement':a,'value':b}).set_index(['measurement'])
 
class points_allocation:
  
    '''
    Method
    ------

    self.fit(X)
    \t Fit the model according to the given initial inputs 
    **Return**
    \t self.X (dataframe)

    self.plot()
    \t Plot Weight of Evidence against Score for each feature
    '''
    def __init__(self, woe_df, coef_df, intercept=0, odds=50, pt_odds=600, pdo=20, decimal=0, 
                 figsize=(6,4.5), color=('#fff200','#aaa69d','#ea2027')):
    
        '''
        Parameters
        ----------

        woe_df : dataframe
        \t =================================================================================
        \t |   | [var] | min | max | bin | [ne] | [e] | [pne]  |  [pe]  |   woe   |   iv   | 
        \t ---------------------------------------------------------------------------------
        \t | 0 |  xxx  | nan | nan |  0  |    0 |   0 | 0.0000 | 0.0000 | -5.7228 | 0.0000 |
        \t | 1 |  xxx  |  0  |  1  |  1  |  651 |   8 | 0.0519 | 0.1951 | -1.3237 | 0.1895 |
        \t | 2 |  xxx  |  1  |  2  |  2  | 1971 |  11 | 0.1572 | 0.2683 | -0.5344 | 0.0594 |
        \t | 3 |  xxx  |  2  |  3  |  3  | 2992 |   7 | 0.2387 | 0.1707 |  0.3350 | 0.0228 |
        \t | 4 |  xxx  |  3  |  4  |  4  | 5318 |  12 | 0.4242 | 0.2927 |  0.3712 | 0.0488 |
        \t | 5 |  xxx  |  4  |  5  |  5  |  641 |   2 | 0.0511 | 0.0488 |  0.0471 | 0.0001 |
        \t | 6 |  xxx  |  5  |  6  |  6  |  963 |   1 | 0.0768 | 0.0244 |  1.1473 | 0.0601 |
        \t =================================================================================
        \t Note: [var] = 'variable', [ne] = 'Non_events', [e] = 'Events', 
        \t       [pne] = 'pct_nonevents', and [pe] = 'pct_events'

        coef_df : (dataframe) 
        \t it is comprised of 'variable' and 'coef' obtained from model

        intercept : float, optional, default: 0
        \t model intercept from logistic regression
        
        odds : int, optional, default: 50
        \t initial odd
        
        pt_odds : int, optional, default: 600
        \t point assigned to initial odd 
        
        pdo : int, optional, default: 20
        \t point that doubles the odd
        
        decimal : int, optional, default: 0
        \t score precision (decimal place)
        
        figsize : tuple of floats, optional, default: (5,4)
        \t width and height of plot (inches)
        
        color : tuple of hex, optional, default: ('#fff200','#aaa69d','#ea2027')
        \t color for event, non-event, and score, respectively

        Return
        ------

        self.adverse_score : float, rejection score
        
        self.min_max : tuple of floats, the possible minimum and maximum scores
        
        self.neutral_score : float, neutral score (weighted average approach)
        
        self.score_df : dataframe, score distribution
        '''
        # pdo = Point Double the Odd
        self.factor = pdo/np.log(2)
        self.offset = pt_odds - self.factor*np.log(odds)
        self.intercept = intercept

        # woe and coefficient datasets
        self.woe_df = woe_df.rename(str.lower, axis='columns')
        self.coef_df = coef_df.rename(str.lower, axis='columns')

        # color attributes
        self.figsize, self.width = figsize, 0.6
        self.posbar_kwargs = dict(color=color[0], label='WOE(Non-event>Event)', alpha=0.8, width=self.width, 
                                  align='center', hatch='////', edgecolor='#4b4b4b', lw=1)
        self.postxt_kwargs = dict(ha='center',va='top', rotation=0, color='#4b4b4b', fontsize=10)
        self.negbar_kwargs = dict(color=color[1], label='WOE(Non-event<Event)', alpha=0.8, width=self.width, 
                                  align='center', hatch='////', edgecolor='#4b4b4b', lw=1)
        self.negtxt_kwargs = dict(ha='center',va='bottom', rotation=0, color='#4b4b4b',fontsize=10)
        self.score_kwargs = dict(color=color[2], lw=1, ls='--', label='score', marker='o',
                                 markersize=4, fillstyle='none')
        self.n_score_kwargs = dict(lw=1.5, ls='-', color=color[2])

        # allocate points, calculate possible max, min, and adverse score
        self.__attribute_score(decimal)
        # neutral socre - weighted average (cutoff)
        self.__neutral_score()

    def __attribute_score(self, decimal=2):

        a = self.woe_df.merge(self.coef_df, on='variable', how='left')
        a = a.loc[a['coef'].notna(),:]

        # scaling calculation
        n_feat = np.unique(a['variable']).shape[0]
        incpt, offset = self.intercept/n_feat, self.offset/n_feat
        score = -(a['coef'].astype(float) * a['woe'] + incpt) * self.factor + offset
        a['score'] = np.round(score.values,decimal)
        self.score_df = a.drop(['coef'], axis=1)

        # possible max and min points
        a = self.score_df[['variable','score']].groupby(['variable']).agg(['min','max']).values
        self.min_max = (sum(a[:,0]), sum(a[:,1]))

        # adverse score (rejection score where woe = 0)
        self.adverse_score = -incpt * self.factor + offset

    def __neutral_score(self):

        a = self.score_df.copy()
        var = a['variable'].unique()[0]
        n_samples = sum(a.loc[a['variable']==var,['non_events','events']].sum())
        a['pct'] = (a['non_events'] + a['events'])/n_samples
        a['n_score'] = a['pct'] * a['score']
        a = a[['variable','n_score']].groupby(['variable']).agg(['sum'])
        columns = ['variable','neutral_score']
        self.neutral_score = pd.DataFrame(a.reset_index().values, columns=columns)

    def fit(self, X):

        '''
        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)

        Return
        ------

        X : (dataframe), shape of (n_samples, n_features + total_score)
        '''
        if isinstance(X, tuple((pd.core.series.Series,pd.DataFrame))):
            self.X = np.zeros(len(X)).reshape(-1,1)
            self.X = pd.DataFrame(data=self.X, columns=['total_score'])
            woe_var = np.unique(self.woe_df['variable'].values)
            columns = [var for var in X.columns if var in woe_var]
            if len(columns) > 0:
                for (n, var) in enumerate(columns):
                    score = pd.DataFrame(data=self.__find_score(X[var]),columns=[var])
                    self.X = self.X.merge(score, left_index=True, right_index=True)
                    self.X['total_score'] = self.X['total_score'] + self.X[var]

    def __find_score(self, X):

        # determine min and max 
        r_min, r_max, x_name = np.nanmin(X), np.nanmax(X), X.name
        a = self.score_df.loc[(self.score_df['variable']==x_name)]
        min_bin = min(np.nanmin(a['min']),r_min)
        max_bin = max(np.nanmax(a['max']),r_max) + 1
        nan_bin = min_bin - 1

        # replace np.nan with the lowest number
        X = pd.Series(X).fillna(nan_bin)

        # create array of bin edges
        bin_edges = a[['min','max']].fillna(nan_bin).values
        bin_edges = np.sort(np.unique(bin_edges.reshape(-1)))
        bin_edges[-1] = max_bin

        # Assign group index to value array and convert to dataframe
        X = np.digitize(X, bin_edges, right=False)
        X = pd.DataFrame(data=X,columns=['bin'])
        X['bin'] = X['bin'] - 1 # Bin in woe_df starts from 0
        X = X.merge(a[['bin','score']], on=['bin'], how='left')
        return X.drop(columns=['bin']).values

    def plot(self, fpath=None, prefix=''):

        '''
        plot score distribution against WOEs 
        
        Parameters
        ----------
        
        fpath : str, optional, default: None
        \t folder path where all plots will be stored
        
        prefix : str, optional, default: ''
        \t prefix that will be added to the beginning of plot's name 
        ''' 
        for var in np.unique(self.score_df['variable']):
            fig, axis = plt.subplots(1, 1, figsize=self.figsize)
            self.__score_woe(axis, var)
            fig.tight_layout()
            if fpath is not None: plt.savefig(fpath + prefix + var + '.png')
            plt.show()

    def __iv_predict(self, iv):

        '''
        Parameter
        ---------

        iv : float, Infomation Value (summation of WOEs)

        Return
        ------

        prediction_strength : (str), IV predictiveness
        '''
        if iv < 0.02: return 'Not useful for prediction'
        elif iv >= 0.02 and iv < 0.1: return 'Weak predictive Power'
        elif iv >= 0.1 and iv < 0.3: return 'Medium predictive Power'
        elif iv >= 0.3: return 'Strong predictive Power'

    def __score_woe(self, axis, var):

        a = self.score_df.loc[self.score_df['variable']==var].copy()
        iv = 'IV = %.4f (%s)' % (a.iv.sum(), self.__iv_predict(a.iv.sum()))
        label = 'Variable: %s \n ' % var
        n_score = self.neutral_score.loc[self.neutral_score['variable']==var]
        n_score = int(n_score['neutral_score'])

        # positive and negative WOEs
        Pos_Y = [max(n,0) for n in a['woe'].values]
        Neg_Y = [min(n,0) for n in a['woe'].values]
        score = a['score'].astype(int).values
        xticks = np.arange(a.shape[0])
        xticklabels = self.__ticklabels(a['min'].values)

        # plots
        bar1 = axis.bar(xticks, Pos_Y, **self.posbar_kwargs)
        bar2 = axis.bar(xticks, Neg_Y, **self.negbar_kwargs)
        tw_axis = axis.twinx()
        lns1 = tw_axis.plot(xticks, score,**self.score_kwargs)
        lns2 = tw_axis.axhline(n_score, **self.n_score_kwargs)
        plots = tuple((bar1, bar2, lns1[0], lns2))
        labels = tuple((bar1.get_label(), bar2.get_label(),
                        'score, in bracket','neutral score (%d)' % n_score))

        for n, s in enumerate(np.array(Pos_Y) + np.array(Neg_Y)):
            if s>0: axis.text(n, -0.05, '{:.2f}\n({:d})'.format(s,score[n]), **self.postxt_kwargs)
            elif s<0: axis.text(n, +0.05, '{:.2f}\n({:d})'.format(s,score[n]) , **self.negtxt_kwargs)

        # format axis
        y_ticks = axis.get_yticks(); diff = np.diff(y_ticks)[0]
        axis.set_ylim(min(y_ticks)-diff,max(y_ticks)+diff)
        axis.set_facecolor('white')
        axis.set_ylabel('Weight of Evidence (WOE)', fontsize=10)
        tw_axis.set_ylabel('Score', fontsize=10)
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels, fontsize=10)
        axis.set_title(label + iva
        kwargs = dict(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, 
                      framealpha=0, edgecolor='none')
        tw_axis.legend(plots, labels, **kwargs)
        axis.grid(False)
  
    def __ticklabels(self, a, decimal=1, amt_lmt=1000):

        '''
        Set Xticklabels format
        '''
        fmt = ['{:.' + str(decimal) + 'f}','{:.' + str(decimal) + 'e}']
        labels = [fmt[0].format(amt) if amt<amt_lmt else fmt[1].format(amt) 
                  for (n,amt) in enumerate(a[1:],1)]
        return ['\n'.join(('missing','(nan)'))] + labels
