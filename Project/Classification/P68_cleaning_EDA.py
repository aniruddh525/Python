# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:08:50 2021

@author: anirudh.kumar.verma
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# load data
cardial=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Project P68\\Myocardial infarction complications.csv")

# study data behaviour
cardial.columns
cardial.info() # all cols are either float or int

cardial.head()

# data description
data_summary=cardial.describe()

# check class variable behaviour (class imbalance)
cardial.LET_IS.value_counts()

## chcek for null values

data_null_count=cardial.isnull().sum() # KFK_BLOOD & IBS_NASL has 1696 and 1628 null values and no impact on class for non null values
# so these two columns can be removed. 

# unique values
cardial.KFK_BLOOD.value_counts()
cardial.IBS_NASL.value_counts()

# drop columns
cardial_1=cardial.drop(['ID','KFK_BLOOD', 'IBS_NASL'], axis=1)

#### evaluate cols having null values


cardial.S_AD_KBRIG.value_counts()
cardial_1.boxplot(column='S_AD_KBRIG') # 6 outliers
cardial.D_AD_KBRIG.value_counts()
cardial_1.boxplot(column='D_AD_KBRIG') # 7
cardial.S_AD_ORIT.value_counts()
cardial_1.boxplot(column='S_AD_ORIT') # 
cardial.D_AD_ORIT.value_counts()
cardial_1.boxplot(column='D_AD_ORIT') # 

cardial_1.boxplot(column='K_BLOOD')
cardial_1['K_BLOOD'].hist()
cardial_1.boxplot(column='NA_BLOOD')
cardial_1['NA_BLOOD'].hist()
cardial_1.boxplot(column='ALT_BLOOD')
cardial_1.boxplot(column='AST_BLOOD')
cardial_1.boxplot(column='L_BLOOD')
cardial_1.boxplot(column='ROE')

# All cols have outliers so mean imputation would be difficult so either remove outliers first 
# or use meadian and most-frequent imputaion 

# finding and removing outliers


def outliers(cardial_11,ft):
    q1 = cardial_11[ft].quantile(0.25)
    q3 = cardial_11[ft].quantile(0.75)
    IQR = q3 - q1
    
    Lower_bound = q1 - 3 * IQR
    Upper_bound = q3 + 3 * IQR
    
# Creating list using lower and upper bound
    Is = cardial_11.index[(cardial_11[ft] < Lower_bound) | (cardial_11[ft] > Upper_bound)]
# 1st condition is any values samller than lower band
# 2nd condition is any values higher than upper band 
# with help of or operator | Either of conditions returns to true its an outlier 
    return Is

# dividing data into 2 dataframes based on data types

data_summary=cardial_1.describe()  

cardial_11=cardial_1[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_12=cardial_1.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)

index_list = []
for column in cardial_11.columns:
    index_list.extend(outliers(cardial_11,column))

index_list


cardial_11.boxplot(column='AGE')
cardial_11.boxplot(column='S_AD_KBRIG')

# unique(index_list)
x = np.array(index_list)
print(len((np.unique(x))))


## Removing Outliers

def remove(cardial_11,Is):
    Is = sorted(set(Is))
    cardial_11 = cardial_11.drop(Is) 
    return cardial_11
df_clean = remove(cardial_11,index_list)
df_clean.shape

## final Df after removing outliers

cardial_2 = pd.concat([df_clean, cardial_12], axis=1, join='inner')

cardial_2.LET_IS.value_counts()

#### Handling null data


# dividing data into 2 dataframes based on imputation startegy to be applied

cardial_2_null_count=cardial_2.isnull().sum() 

grp_0 = cardial_2[cardial_2['LET_IS']==0]
grp_1 = cardial_2[cardial_2['LET_IS']==1]
grp_2 = cardial_2[cardial_2['LET_IS']==2]
grp_3 = cardial_2[cardial_2['LET_IS']==3]
grp_4 = cardial_2[cardial_2['LET_IS']==4]
grp_5 = cardial_2[cardial_2['LET_IS']==5]
grp_6 = cardial_2[cardial_2['LET_IS']==6]
grp_7 = cardial_2[cardial_2['LET_IS']==7]



cardial_210=grp_0[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_220=grp_0.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)

cardial_211=grp_1[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_221=grp_1.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)

cardial_212=grp_2[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_222=grp_2.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)


cardial_213=grp_3[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_223=grp_3.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)

cardial_214=grp_4[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_224=grp_4.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)

cardial_215=grp_5[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_225=grp_5.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)

cardial_216=grp_6[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_226=grp_6.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)

cardial_217=grp_7[["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE']]

cardial_227=grp_7.drop(["AGE","S_AD_KBRIG",'D_AD_KBRIG','S_AD_ORIT','D_AD_ORIT',
                      'K_BLOOD','NA_BLOOD','ALT_BLOOD','AST_BLOOD','L_BLOOD',
                      'ROE'], axis=1)


### imputation function

def impute(int_fea,cat_fea):
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    integer_features = list(int_fea.columns)
    cat_features = list(cat_fea.columns)
    integer_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median'))])
    cat_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent'))])

    preprocessor_int = ColumnTransformer(
        transformers=[
            ('ints', integer_transformer, integer_features)])

    preprocessor_cat = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_features)])

    cardial_int_imputed=preprocessor_int.fit_transform(int_fea)
    cardial_cat_imputed=preprocessor_cat.fit_transform(cat_fea)
    cardial_int_df = pd.DataFrame(cardial_int_imputed, columns=int_fea.columns)
    cardial_cat_df = pd.DataFrame(cardial_cat_imputed, columns=cat_fea.columns)

    cardial_f = pd.concat([cardial_int_df, cardial_cat_df], axis=1, join='inner')
    return cardial_f


cardial_20 = impute(cardial_210,cardial_220)
cardial_21 = impute(cardial_211,cardial_221)
cardial_22 = impute(cardial_212,cardial_222)
cardial_23 = impute(cardial_213,cardial_223)
cardial_24 = impute(cardial_214,cardial_224)
cardial_25 = impute(cardial_215,cardial_225)
cardial_26 = impute(cardial_216,cardial_226)
cardial_27 = impute(cardial_217,cardial_227)

cardial_3 = pd.concat([cardial_20,cardial_21,cardial_22,cardial_23,cardial_24,cardial_25,
                       cardial_26,cardial_27])

# chcek for null values 
cardial_3_nullcount=cardial_3.isnull().sum() # no null data


### Feature engg/feature selection


correlation_matrix=cardial_3.corr()
LET_corr=correlation_matrix['LET_IS']

# RAZRIV	0.3502529961666059
# ZSN_A	0.17699835306173758
# FIBR_JELUD	0.17312539446015654
# NITR_S	0.16535975148816462
# DLIT_AG	0.16164087174946687
# AGE	0.15159285926098678
# ant_im	0.15149508574348397
# STENOK_AN	0.1465350222671996
# nr_04	0.13959782282369676
# MP_TP_POST	0.1328904264294872


correlation_matrix_LET_IS=correlation_matrix[correlation_matrix.LET_IS>0.1]
# correlation is not giving much info as the max is 0.3 between any 
#input variable and class variable

#drawing scatterplot on numerical data
sns.pairplot(cardial_3)


# 1st method univariate feature engg

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

array = cardial_3.values
X = array[:,0:120]
Y = array[:,120]
# feature extraction
test = SelectKBest(score_func=chi2, k=10) # k tells how many top features we need
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
scores=fit.scores_
scores_df=pd.DataFrame(scores)

top_index=scores_df.sort_values(0,ascending = False).head(10).index
colname_uni = cardial_3.columns[[top_index]]
# features = fit.transform(X)

	
# 0	RAZRIV
# 1	S_AD_KBRIG
# 2	ROE
# 3	ZSN_A
# 4	DLIT_AG
# 5	S_AD_ORIT
# 6	STENOK_AN
# 7	AGE
# 8	ant_im
# 9	K_SH_POST


# 2nd method RFE

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction
model = LogisticRegression(max_iter=400,solver='liblinear')
rfe = RFE(model, 10)
fit_rfe= rfe.fit(X, Y)


#Num Features: 
fit_rfe.n_features_

#Selected Features:
fit_rfe.support_

# Feature Ranking:
rankings=pd.DataFrame(fit_rfe.ranking_)


colname_rfe = cardial_3.columns[[rankings.sort_values(0,ascending = False).head(10).index]]

	
# 0	nr_07
# 1	n_p_ecg_p_05
# 2	n_r_ecg_p_09
# 3	np_09
# 4	nr_08
# 5	S_AD_ORIT
# 6	fibr_ter_07
# 7	D_AD_ORIT
# 8	ROE
# 9	AGE



# 3rd method , Decision tree based

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, Y)
feature_imp=pd.DataFrame(model.feature_importances_)

colname_DTree = cardial_3.columns[[feature_imp.sort_values(0,ascending = False).head(10).index]]

	
# 0	RAZRIV
# 1	S_AD_KBRIG
# 2	IBS_POST
# 3	D_AD_KBRIG
# 4	STENOK_AN
# 5	AST_BLOOD
# 6	L_BLOOD
# 7	AGE
# 8	TIME_B_S
# 9	DLIT_AG



# 4th method - Boruta plot

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
# find all relevant features - 5 features should be selected
feat_selector.fit(X, Y)
# check selected features - first 5 features are selected
feat_selector.support_
# check ranking of features
Feature_boruta=pd.DataFrame(feat_selector.ranking_)
colname_boruta = cardial_3.columns[[Feature_boruta.sort_values(0,ascending = False).head(10).index]]
# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
#To get the new X_train now with selected features
#X_filtered.columns[feat_selector.support_]

	
# 0	n_p_ecg_p_01
# 1	np_07
# 2	nr_08
# 3	fibr_ter_07
# 4	nr_07
# 5	n_r_ecg_p_08
# 6	n_r_ecg_p_09
# 7	n_p_ecg_p_05
# 8	np_09
# 9	fibr_ter_05



## 5th method pps

# import seaborn as sns
# import ppscore as pps

# def heatmap(df):
#     ax = sns.heatmap(df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
#     ax.set_title('PPS matrix')
#     ax.set_xlabel('feature')
#     ax.set_ylabel('target')
#     return ax


# def corr_heatmap(df):
#     ax = sns.heatmap(df, vmin=-1, vmax=1, cmap="BrBG", linewidths=0.5, annot=True)
#     ax.set_title('Correlation matrix')
#     return ax

# pps_matrix=pps.matrix(cardial_3)

# all_LETIS=pps_matrix[pps_matrix.y == 'LET_IS']

# cardial_3.iloc[:,115]



## Selecting Decision tree as final method for Feature selection

cardial_4=cardial_3[colname_DTree]
cardial_4['LET_IS']=cardial_3['LET_IS']

### handling class imbalance problem

cardial_4.LET_IS.value_counts()

# 0.0    1360
# 1.0      68
# 3.0      51
# 6.0      24
# 7.0      22
# 4.0      21
# 2.0      17
# 5.0      10

X=cardial_4.iloc[:,:-1]
Y=cardial_4['LET_IS']

from imblearn.over_sampling import SMOTE
from collections import Counter
Counter(Y)

smt = SMOTE(random_state=42)
X_res, Y_res = smt.fit_resample(X,Y)
Counter(Y_res)

cardial_final=pd.concat([X_res, Y_res], axis=1, join='inner')
cardial_final.to_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Project P68\\cardial_final.csv")
### Model building


from sklearn.model_selection import train_test_split
IV_train,IV_test,DV_train,DV_test= train_test_split(X_res,Y_res, test_size=0.3, random_state=100)


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

svc_classifier = SVC(kernel='linear')
svc_classifier.fit(IV_train,DV_train)

# svc_classifier = cross_val_score(estimator=svc_classifier,X=train_set2_features,y=train_set2_traget,cv=10)


y_pred=svc_classifier.predict(IV_test)
np.mean(y_pred==DV_test) ## 0.85

# xyz

from sklearn.ensemble import RandomForestClassifier
rfc_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                        random_state=101, oob_score=True,
                                        max_features=None, min_samples_leaf=30)
rfc_classifier.fit(IV_train,DV_train)

data_pred = rfc_classifier.predict(IV_test)
np.mean(DV_test==data_pred) # 0.91

from sklearn.metrics import confusion_matrix
confusion_matrix(DV_test, data_pred)

from sklearn.metrics import classification_report
# from sklearn.metrics import roc_curve

print(classification_report(DV_test, data_pred))



from sklearn.naive_bayes import GaussianNB,MultinomialNB

gnb_clf = GaussianNB()
gnb_clf.fit(IV_train,DV_train)
test_pred = gnb_clf.predict(IV_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(DV_test,test_pred)
ac = accuracy_score(DV_test,test_pred) # 0.53
cm,ac

mnb_clf = MultinomialNB()
mnb_clf.fit(IV_train,DV_train)
test_pred = mnb_clf.predict(IV_test)
cm = confusion_matrix(DV_test,test_pred)
ac = accuracy_score(DV_test,test_pred) # 0.58
cm,ac



from sklearn.neural_network import MLPClassifier
mlp_classifier=MLPClassifier(hidden_layer_sizes=(8,4),max_iter=2000,alpha=0.00001,solver='adam',verbose=0,random_state=21,tol=0.000001)
mlp_classifier.fit(IV_train,DV_train)

test_pred=mlp_classifier.predict(IV_test)

cm = confusion_matrix(DV_test,test_pred)
ac = accuracy_score(DV_test,test_pred) # 0.72
cm,ac



from sklearn.tree import DecisionTreeClassifier
DT_cls = DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_split=4)
DT_cls.fit(IV_train,DV_train)
test_pred=DT_cls.predict(IV_test)
np.mean(test_pred==DV_test) ## 0.95

cm = confusion_matrix(DV_test,test_pred)
ac = accuracy_score(DV_test,test_pred) # 0.95
cm,ac



import pickle
# save the model to disk
filename = 'DecisionTreeClassifier_Model.sav'
pickle.dump(DT_cls, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(IV_test, DV_test)
result















