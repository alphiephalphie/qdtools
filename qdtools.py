'''
The :mod: 'qdtools' streamlines many steps in data cleaning and
'''



# Author: Alphonso Woodbury <alphonsorees@gmail.com>
#Thanks to Ammar Ali and Marisa Mitchell
#
# License: BSD 3 clause

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('float_format', '{:f}'.format)

class preprocessingVanilla():
    '''
    LinearVanilla will help you quickly analyze a dataset for its potential in Linear, Ridge, or Lasso models.\nThe module is primed for datasets that have already been cleaned, including removing NaN values.'

    """
    ...

    Read more in the :ref:`User Guide <??>`.
    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    Notes
    -----

    References
    ----------

    """

    '''

    def __init__(self):
        print('Object instantiated. Use help() method for guidance.')

    def help(self):
        print('1: load_data()\n2: target_and_split()\n3: set_cat()\n\nModels:\nsm_linear() - standard linear regression using statmodels\nqd_LRL() - sklearn LinearRegression, Ridge, and Lasso \n\n')

    def load_data(self,path):
        self.df = pd.read_csv(path)

        self.uniques = {}
        for col in self.df.columns:
            ocounts = self.df[col].value_counts()
            self.uniques.update({col : {'unique' :self.df[col].nunique(),'1_count':ocounts[ocounts == 1].count(),'dtype' : self.df[col].dtype, 'null':self.df[col].isna().sum()}})

        self.hud = pd.DataFrame.from_dict(self.uniques,dtype='int',orient='index',columns=['unique','1_count','dtype','null'])
        self.hud['percent'] = (self.hud.unique / self.df.count())*100
        self.hud['recommend_categorical'] = None
        self.hud['p_value'] = None
        self.hud['recommend_drop'] = None
        self.cat_feats = []
        self.cont_feats = []
        self.X_train = pd.DataFrame()
        print(path + ' loaded into ''df'' dataframe. Use qd_info_df() for detailed information.')
        self.fitted = False

    def target_and_split(self,target,test_size=0.25,random_state=None):
        self.y = self.df[target]
        self.X = self.df.drop(columns=[target],axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state = random_state, test_size = test_size)
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        return self.X_train.shape, self.y_train.shape

    def drop(self,*columns):
        if self.X_train.empty:
            print('Cannot drop from original dataframe. Please use target_and_split() first.')
        else:
            for column in columns:
                self.X_train = self.X_train.drop(columns=column, axis=1)
                self.X_test = self.X_test.drop(columns=column, axis=1)

    def set_cat(self,*features):
        if self.X_train.empty:
            print('Cannot change original dataframe. Please use target_and_split() first.')
        else:
            for feature in features:
                self.X_train[feature] = self.X_train[feature].astype('object') # change to X_train
                self.X_test[feature] = self.X_test[feature].astype('object') # change to X_train
                self.cat_feats.append(feature)
        self.update_hud()

    def replace_nulls(self,*features,replacer):
        for feature in features:
            self.X_train[feature] = self.X_train[feature].apply(lambda x: x if not pd.isnull(x) else replacer)
        self.update_hud()

    def qd_info_df(self):
        '''
        ...
        '''
        self.df.head()
        self.df.info(verbose=True,null_counts=True)
        return self.df.describe(percentiles=[.01,.1,.25,.5,.75,.9,.99],include='all')

    def head(self):
        return self.df.head(10)

    def cut99(self,*features):
        for feature in features:
            self.feat99 = self.df[feature].quantile(.99)
            self.df = self.df[self.df[feature] < self.feat99]

    def update_hud(self):
        for col in self.X_train.columns:
            self.uniques.update({col : {'unique' :self.X_train[col].nunique(),'dtype' : self.X_train[col].dtype, 'null':self.X_train[col].isna().sum()}})
        self.hud = pd.DataFrame.from_dict(self.uniques,dtype='int',orient='index',columns=['unique','1_count','dtype','null'])
        self.hud['unique'] = pd.DataFrame.from_dict(self.uniques,dtype='int',orient='index',columns=['unique'])
        self.hud['dtype'] = pd.DataFrame.from_dict(self.uniques,dtype='int',orient='index',columns=['dtype'])
        self.hud['percent'] = (self.hud.unique / self.X_train.count())*100
        #for self.feature_analysis['dtype'] == 'object':
        self.hud['recommend_categorical'] = self.hud['dtype'] == 'object'

    def view_hud(self):
        if self.X_train.empty:
            print('Please use target_and_split() to view HUD.')
        else:
            return self.hud

class LinearVanilla(preprocessingVanilla):
    def dummyfittrans(self):
        scale = StandardScaler()

        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        if len(self.cat_feats) > 0:
            self.train_dummies = pd.get_dummies(self.X_train[self.cat_feats],drop_first=True)
            self.X_train = self.X_train.drop(columns=self.cat_feats)
            self.X_train = self.X_train.join(self.train_dummies,how='left')


            self.test_dummies = pd.get_dummies(self.X_test[self.cat_feats],drop_first=True)
            #self.X_test = self.X_test.drop(columns=self.cat_feats)
            #self.X_test = self.X_train.join(self.test_dummies,how='left')

        self.X_train_cont   = self.X_train
        self.X_train = pd.DataFrame(scale.fit_transform(self.X_train_cont), columns=self.X_train_cont.columns)
        self.fitted = True

    def smlinear(self):
        if not self.fitted:
            self.dummyfittrans()

        self.X_train_const = sm.add_constant(self.X_train)
        self.model = sm.OLS(self.y_train, self.X_train_const, hascont=True)
        self.fitted_model = self.model.fit()
        self.summary = self.fitted_model.summary()
        self.pdf = pd.read_html(self.summary.tables[1].as_html(),header=0,index_col=0)[0]

    def linreg(self):
        lin = LinearRegression()
        lin.fit(self.X_train, self.y_train)
        self.lrscore = lin.score(self.X_train,self.y_train)

    def ridge(self,ridge_alpha=1):
        ridge = Ridge(alpha=ridge_alpha,max_iter=1000)
        ridge.fit(self.X_train, self.y_train)
        self.ridgescore = ridge.score(self.X_train, self.y_train)

    def lasso(self,lasso_alpha=1):
        lasso = Lasso(alpha=lasso_alpha,max_iter=1000)
        lasso.fit(self.X_train,self. y_train)
        self.lassoscore = lasso.score(self.X_train, self.y_train)

    def qd_LRL(self,ridge_alpha=1,lasso_alpha=1):
        self.linreg()
        self.ridge(ridge_alpha=ridge_alpha)
        self.lasso(lasso_alpha=lasso_alpha)
        self.lrlscores = {}
        self.lrlscores.update({'LinReg' :self.lrscore,'Ridge':self.ridgescore,'Lasso' : self.lassoscore})
        self.scoresheet = pd.DataFrame.from_dict(self.lrlscores,orient='index',columns=['Scores'])#,columns=['LinReg','Ridge','Lasso'])

class LogisticVanilla():
    pass
