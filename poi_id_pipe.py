#!/usr/bin/python

import sys
import pickle
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, FeatureUnion
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
				'from_messages','to_messages'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s) 
## for this first run I wont be creating any new feature before fitting
### Store to my_dataset for easy export below.
my_dataset = data_dict

def mod_my_dataset_removing_keys_with_nan_values_in_the_ft_list (feature_list):
    to_del = []
    for person in my_dataset:
        for feature in features_list[1:]:
                if my_dataset[person][feature] == "NaN":
                    to_del.append(person)
                    #pprint.pprint(set(to_del))


    for key in set(to_del):
        my_dataset.pop(key)

    return my_dataset


## modify my_dataset to remove all the NaN 
#my_dataset = mod_my_dataset_removing_keys_with_nan_values_in_the_ft_list(features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True) #data here is my_dataset light : it just keeps the features im interested in
labels, features = targetFeatureSplit(data) # divide data in labels & features 
print 'len(labels),len(features) is',len(labels),'\n'


# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=7)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(features, labels).transform(features)

GNB = GaussianNB()

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("gnb", GNB)])

param_grid = dict(features__pca__n_components=[1, 2],
                  features__univ_select__k=[2, 3,4,5,6,7],
                  )

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(features, labels)
print(grid_search.best_estimator_)
clf = grid_search.best_estimator_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)