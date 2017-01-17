#!/usr/bin/python

import sys
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import pprint 
pp = pprint.PrettyPrinter(indent=4)
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','restricted_stock',\
        "long_term_incentive","total_stock_value",\
        'expenses','other','total_payments','deferral_payments',\
        'deferred_income','exercised_stock_options','loan_advances',\
        'bonus',\
        'from_messages','to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s) 
## for this first run I wont be creating any new feature before fitting
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True) #data here is my_dataset light : it just keeps the features im interested in
labels, features = targetFeatureSplit(data) # divide data in labels & features 
print 'len(labels),len(features) is',len(labels),'\n'


def run_GNB(quant_of_params,features,labels,features_list):

    ### Task 3.5: Features selection
    selector = SelectKBest(f_classif, k=quant_of_params)
    selectedFeatures = selector.fit(features, labels)
    selected_features_list = [features_list[i+1] for i in selectedFeatures.get_support(indices=True)]
    features_list = features_list[:1]+selected_features_list
    print 'New feature_list after SelectKbest is\n',features_list,'\n'

    ### Extract features and labels from dataset for local testing 
    data = featureFormat(my_dataset, features_list, sort_keys = True) #data here is my_dataset light : it just keeps the features im interested in
    labels, features = targetFeatureSplit(data) # divide data in labels & features 
    print 'quantity of data points is',len(features)


    #------ GNB
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features,labels)


    from tester_mod import test_classifier
    test_classifier(clf, my_dataset, features_list)

for k in range(2,10):
    print k,'features'
    run_GNB(k,features,labels,features_list)
    print '---------\n'




