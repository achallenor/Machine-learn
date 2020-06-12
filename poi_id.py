
# coding: utf-8

# In[1]:

# Import required modules
import sys
import pickle
import pandas as pd
import numpy as np
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:

# Load the dictionary into a dataframe and examine it
dataset_df=pd.DataFrame.from_dict(data_dict,orient='index')


# In[4]:

# Replace 'NaN' string with Null (NaN)
dataset_df.replace('NaN',np.nan,inplace=True)


# In[5]:

# Dataset details
num_employees=len(dataset_df)
num_poi=len(dataset_df[dataset_df['poi']==True])
num_non_poi=num_employees-num_poi
num_vals=num_employees-dataset_df.isnull().sum()


# In[6]:

# Selected features
POI_label=['poi'] # Boolean, represented as integer
financial_features=['salary','deferral_payments','total_payments','bonus','deferred_income',
 'total_stock_value','expenses','exercised_stock_options','other',
 'long_term_incentive','restricted_stock'] # Units are in US dollars
email_features=['to_messages','from_poi_to_this_person','from_messages',
 'from_this_person_to_poi','shared_receipt_with_poi'] # Units are number of
# We will ignore:
# 'email_address' - not numerical data
# 'restricted_stock_deferred' and 'director_fees' - less than 10% data for POI
# 'loan_advances' - less than 10% data
features_list=(POI_label+financial_features+email_features)
#print 'Number of initial features: ',len(features_list)


# In[7]:

#Drop email address since we are not using it in this analysis
dataset_df.drop('email_address',axis=1,inplace=True)


# In[8]:

# Drop the following:
# TOTAL - Spreadsheet aggregation included by mistake (outlier)
# LOCKHART EUGENE E - Does not contain any numerical data
# THE TRAVEL AGENCY IN THE PARK - Not an individual (Alliance Worldwide - co-owned by the s
dataset_df.drop(['TOTAL','LOCKHART EUGENE E','THE TRAVEL AGENCY IN THE PARK'],axis=0,inplace=True)


# In[9]:

# New financial features:
dataset_df['fraction_bonus_salary']=dataset_df['bonus']/dataset_df['salary']
dataset_df['fraction_bonus_total']=dataset_df['bonus']/dataset_df['total_payments']
dataset_df['fraction_salary_total']=dataset_df['salary']/dataset_df['total_payments']
dataset_df['fraction_stock_total']=dataset_df['total_stock_value']/dataset_df['total_payments']


# In[10]:

# New email features:
dataset_df['fraction_to_poi']=dataset_df['from_this_person_to_poi']/dataset_df['from_messages']
dataset_df['fraction_from_poi']=dataset_df['from_poi_to_this_person']/dataset_df['to_messages']


# In[11]:

# Add new features to feature list
new_features_list=['fraction_bonus_salary',
 'fraction_bonus_total',
'fraction_salary_total',
'fraction_stock_total',
'fraction_to_poi',
'fraction_from_poi']
extended_features_list=features_list+new_features_list
#print 'Number of extended features: ',len(extended_features_list)


# In[12]:

# Cleaned and trimmed dataset
num_employees=len(dataset_df[extended_features_list])
num_poi=len(dataset_df[dataset_df['poi']==True])
num_non_poi=num_employees-num_poi
num_vals=num_employees-dataset_df[extended_features_list].isnull().sum()


# In[13]:

# Replace Null (NaN) entries with 0.0 to prevent errors in algorithms
dataset_df.fillna(value=0.0,inplace=True)


# In[14]:

### Store to my_dataset for easy export below.
my_dataset=dataset_df.to_dict('index')
### Extract original features and labels from dataset
data=featureFormat(my_dataset,features_list,sort_keys=True)
labels,features=targetFeatureSplit(data)
### Extract extended features and labels from dataset
data=featureFormat(my_dataset,extended_features_list,sort_keys=True)
labels2,features2=targetFeatureSplit(data)


# In[15]:

# Select K-Best features
n=12

k_best1=SelectKBest(score_func=f_classif,k=n)
k_best1.fit(features,labels)
feature_scores1=zip(features_list[1:],k_best1.scores_)
k_best_features1=OrderedDict(sorted(feature_scores1,key=lambda x: x[1],reverse=True))

k_best2=SelectKBest(score_func=f_classif,k=n)
k_best2.fit(features2,labels2)
feature_scores2=zip(extended_features_list[1:],k_best2.scores_)
k_best_features2=OrderedDict(sorted(feature_scores2,key=lambda x: x[1],reverse=True))


# In[16]:

# Fit Decision Tree to unscaled features and get feature importances
clf1=DecisionTreeClassifier()
clf1=clf1.fit(features2,labels2)
feature_importances1=zip(extended_features_list[1:],clf1.feature_importances_)
important_features1=OrderedDict(sorted(feature_importances1,key=lambda x: x[1],reverse=True))

# Scale features
scaler=StandardScaler(copy=True)
scaled_features=scaler.fit_transform(features2)

# Fit Decision Tree to scaled features and get feature importances
clf2=DecisionTreeClassifier()
clf2=clf2.fit(scaled_features,labels2)
feature_importances2=zip(extended_features_list[1:],clf2.feature_importances_)
important_features2=OrderedDict(sorted(feature_importances2,key=lambda x: x[1],reverse=True))


# In[17]:

def classify(clf):
    test_classifier(clf,my_dataset,extended_features_list)


# In[18]:

#GaussianNB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf,my_dataset,extended_features_list)


# In[19]:

#Random Forest Classifier (Unscaled)
pipeline=Pipeline([('kbest',SelectKBest()),
                   ('clf',RandomForestClassifier())])
clf=pipeline.fit(features2,labels2)
classify(clf)


# In[20]:

#K Neighbors Classifier (Unscaled)
pipeline=Pipeline([('kbest',SelectKBest()),
                 ('clf',KNeighborsClassifier())])
clf=pipeline.fit(features2,labels2)
classify(clf)


# In[21]:

#K Neighbors Classifier (Scaled)
pipeline=Pipeline([('scaler',StandardScaler()),
                 ('kbest',SelectKBest()),
                 ('clf',KNeighborsClassifier())])
clf=pipeline.fit(features2,labels2)
classify(clf)


# In[22]:

#Support Vector Classifier (Scaled)
pipeline=Pipeline([('scaler',StandardScaler()),
                 ('kbest',SelectKBest()),
                 ('clf',SVC(kernel="linear"))])
clf=pipeline.fit(features2,labels2)
classify(clf)


# In[23]:

#AdaBoost Classifier (Unscaled)
pipeline=Pipeline([('kbest',SelectKBest()),
                   ('clf',AdaBoostClassifier())])
clf=pipeline.fit(features2,labels2)
classify(clf)


# In[24]:

#DecisionTree Classifier
pipeline=Pipeline([('scaler',StandardScaler()),
                   ('kbest',SelectKBest()),
                   ('clf',DecisionTreeClassifier())])
param_grid=([{'kbest__k':[6,12,18],
              'clf__max_depth':[None,1,2],
              'clf__min_samples_split':[10,20,30],
              'clf__class_weight':[None,'balanced']}])
clf=GridSearchCV(pipeline,param_grid,scoring='f1').fit(features2,labels2).best_estimator_
#perf_labels,perf_metrics=classify(clf)
classify(clf)


# In[25]:

#AdaBoost Classifier
pipeline=Pipeline([('kbest',SelectKBest()),
                   ('clf',AdaBoostClassifier())])
param_grid=([{'kbest__k':[6,12,18],
              'clf__base_estimator':[DecisionTreeClassifier(class_weight='balanced',max_depth=1),
                                     DecisionTreeClassifier(class_weight='balanced',max_depth=2)],
              'clf__n_estimators':[25,50,75],
              'clf__learning_rate':[0.01,0.1,1.0],
              'clf__algorithm':['SAMME']}])
clf=GridSearchCV(pipeline,param_grid,scoring='f1').fit(features2,labels2).best_estimator_
#perf_labels,perf_metrics=classify(clf)
classify(clf)


# In[26]:

#Store this classifier
CLF=clf
final_kbest=CLF.named_steps['kbest']
final_clf=CLF.named_steps['clf']
#final_perf_labels=perf_labels
#final_perf_metrics=perf_metrics


# In[27]:

CLF.named_steps['clf']


# In[28]:

#K Neighbors Classifier (Unscaled)
pipeline=Pipeline([('kbest',SelectKBest()),
                   ('clf',KNeighborsClassifier())])
param_grid=([{'kbest__k':[6,12,18],
              'clf__n_neighbors':[3,4,5]}])
clf=GridSearchCV(pipeline,param_grid,scoring='f1').fit(features2,labels2).best_estimator_
#perf_labels,perf_metrics=classify(clf)
classify(clf)


# In[29]:

#Select K-Best
n=final_kbest.k
k_best=final_kbest
k_best.fit(features2,labels2)
feature_scores=zip(extended_features_list[1:],k_best.scores_)
k_best_features=OrderedDict(sorted(feature_scores,key=lambda x: x[1]))
#AdaBoost Classifier
clf=final_clf
clf=clf.fit(k_best.transform(features2),labels2)
feature_importances=zip(extended_features_list[1:],clf.feature_importances_)
important_features=OrderedDict(sorted(feature_importances,key=lambda x: x[1]))


# In[30]:

#Output Classifier Parameters
#print 'Best performing classifier: '+str(final_clf)[:str(final_clf).find('(')]+'\n'
#for k in CLF.named_steps.values():
#    print k


# In[31]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(CLF,my_dataset,extended_features_list)


# In[ ]:



