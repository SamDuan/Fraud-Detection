#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn import tree
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydotplus
import collections

###########################################
### Step 1: Select features of interest ###
###########################################

# Build the original features list with 19 features
features_list = ['poi',
                 'to_messages', 
                 'from_poi_to_this_person',
                 'from_messages', 
                 'from_this_person_to_poi', 
                 'shared_receipt_with_poi', 
                 'salary', 
                 'deferral_payments', 
                 'total_payments', 
                 'loan_advances', 
                 'bonus', 
                 'restricted_stock_deferred',
                 'deferred_income', 
                 'total_stock_value', 
                 'expenses', 
                 'exercised_stock_options',
                 'long_term_incentive', 
                 'restricted_stock', 
                 'director_fees'] 

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###############################
### Step 2: Remove outliers ###
###############################
    
# Visualize data to identify the outliers - 1st try 
data = featureFormat(data_dict, features_list) # data is a numPy array

salary = data[:,features_list.index('salary')]
bonus = data[:,features_list.index('bonus')] 

i = 0
while i < len(bonus):
    if data[i, 0] == 1:
        plt.scatter(salary[i], bonus[i], color='red')
    else:
        plt.scatter(salary[i], bonus[i], color='blue')
    i+=1
    
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

key_list = [k for k in data_dict.keys() \
            if data_dict[k]["salary"] != 'NaN' and \
            data_dict[k]["salary"] > 2.5e7]
data_dict.pop("TOTAL", 0)

# Visualize data to identify the outliers - 2nd try 
data = featureFormat(data_dict, features_list) # data is a numPy array
salary = data[:,features_list.index('salary')]
bonus = data[:,features_list.index('bonus')] 

i = 0
while i < len(bonus):
    if data[i, 0] == 1:
        plt.scatter(salary[i], bonus[i], color='red')
    else:
        plt.scatter(salary[i], bonus[i], color='blue')
    i+=1
    
plt.xlabel("salary")
plt.ylabel("bonus")
plt.axvline(x = 1e6)
plt.axhline(y = 5e6)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.show()

# Use a list comprehension to find outliers
key_list_2 = [k for k in data_dict.keys() if data_dict[k]["poi"] == False \
              and data_dict[k]["salary"] != 'NaN' \
              and data_dict[k]["bonus"] != 'NaN' \
              and (data_dict[k]["salary"] > 1e6 \
              or data_dict[k]["bonus"] > 5e6)]

# The removal of outliers, key_list_2
data_dict.pop("LAVORATO JOHN J", 0)
data_dict.pop("FREVERT MARK A", 0)    

#####################################
### Step 3: Create new feature(s) ###
#####################################
              
# Define a function generating the new features              
def computeFraction(poi_messages, all_messages):
    if poi_messages == "NaN" or all_messages == "NaN" or all_messages == 0:
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)
    return fraction              

# Update data_dict with 2 new features 
# Plot the 2 new features and color POI as red, no-POI as blue
for name in data_dict:

    data_point = data_dict[name]

    from_poi = data_point["from_poi_to_this_person"]
    to_all = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi, to_all)
    data_point["fraction_from_poi"] = fraction_from_poi

    to_poi = data_point["from_this_person_to_poi"]
    from_all = data_point["from_messages"]
    fraction_to_poi = computeFraction(to_poi, from_all)
    data_point["fraction_to_poi"] = fraction_to_poi              
    
    if data_point["poi"]:
        plt.scatter(fraction_from_poi, fraction_to_poi, color='red')
    else:
        plt.scatter(fraction_from_poi, fraction_to_poi, color='blue')              

plt.xlabel("Percentage of emails from POIs to this person (%)")
plt.ylabel("Percentage of emails from this person to POIs (%)")
plt.show()

# Update features_list with 2 new features. Now 21 features included.
features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')

# 
key_list_3 = [k for k in data_dict.keys() if data_dict[k]["poi"] == False \
              and (data_dict[k]["fraction_from_poi"] > 0.15 \
              or data_dict[k]["fraction_to_poi"] > 0.8)]

# Data preparation for feature selection by SelectKBest
mydata = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(mydata)
features = map(abs, features)

# Calculate and plot the score (F-value) for each feature
kb = SelectKBest(f_classif, k='all').fit(features, labels)
scores = kb.scores_[kb.get_support()]

dfn = pd.DataFrame(data=features_list[1:])
dfn[1] = scores
dfn.columns=['Features','Scores']
dfn = dfn.sort_values(by=['Scores'])
dfn.set_index('Features').plot(grid = 'TRUE', kind='barh')

###################################################
### Step 4: Set up the classifiers and pipeline ###
###################################################

# Split the data into a training data set and a testing data set
features_train, features_test, labels_train, labels_test = train_test_split(
        features, 
        labels, 
        test_size=0.3, 
        random_state=42)

# A prelim run using decision tree without any optimization, 
# setting as a reference point
dtree = tree.DecisionTreeClassifier()
dtree.fit(features_train, labels_train)
print 'A prelim run - tree and the result is:'
test_classifier(dtree, data_dict, features_list, folds = 1000)

# A prelim run using logistic regression without any optimization,
# setting as as reference point
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(features_train, labels_train)
print 'A prelim run - lr and the result is:'
test_classifier(lr, data_dict, features_list, folds = 1000)


# Set up a pipeline to find out the optimal number of features to include
skb = SelectKBest()
dtree = tree.DecisionTreeClassifier()
pipe =  Pipeline(steps=[("SKB", skb), ("dt", dtree)])
param_grid = {  # Rules for naming variables:
                # the quoted name in pipeline + 2 underscores + the parameter:
                "SKB__k": range(1, len(features_list)-1),
              }
sss = StratifiedShuffleSplit(random_state=0)
dtclf = GridSearchCV(pipe, param_grid, scoring = 'f1', cv = sss)

#####################################################
### Step 5: Fit, tune and evaluate the classifier ###
#####################################################
dtclf.fit(features, labels)
clf = dtclf.best_estimator_
clf.fit(features, labels)

features_selected_bool = dtclf.best_estimator_.named_steps['SKB'].get_support()
features_selected_list = np.array(features_list[1:])[features_selected_bool]

# note that features_list need to be updated before use below
print 'Optimal # of features and the result is:'
result = test_classifier(clf, 
                         data_dict, 
                         np.insert(features_selected_list, 0, 'poi'), 
                         folds = 1000)

### Visualize the result
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [0.82713, 0.85667]
bars2 = [0.34533, 0.46265]
bars3 = [0.33100, 0.46450]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, 
        bars1, 
        color='#BE6E46', 
        width=barWidth, 
        edgecolor='white', 
        label='Accuracy')
plt.bar(r2, 
        bars2, 
        color='#A1869E', 
        width=barWidth, 
        edgecolor='white', 
        label='Precision')
plt.bar(r3, 
        bars3, 
        color='#59594A', 
        width=barWidth, 
        edgecolor='white', 
        label='Recall')
 
# Add xticks on the middle of the group bars
plt.xlabel('Progression of optimization')
plt.xticks([r + barWidth for r in range(len(bars1))], 
            ['21 features', '10 features'])
plt.ylabel('A.U.')
plt.ylim(0, 1)
plt.axhline(y = 0.3)
 
# Create legend & Show graphic
plt.legend()
plt.show()

# More parameters tuning on the decision tree
skb = SelectKBest(k=10)
dtree = tree.DecisionTreeClassifier()
pipe =  Pipeline(steps=[("SKB", skb), ("dt", dtree)])
param_grid = {  # Rules for naming variables:
                # the quoted name in pipeline + 2 underscores + the parameter:
                "dt__criterion": ["gini", "entropy"],
                "dt__min_samples_split": [2, 10, 20],
                "dt__max_depth": [None, 2, 5, 10],
                "dt__min_samples_leaf": [1, 5, 10],
                "dt__max_leaf_nodes": [None, 5, 10, 20],
                }

sss = StratifiedShuffleSplit(random_state=0)
dtcclf = GridSearchCV(pipe, param_grid, scoring = 'f1', cv = sss)

dtcclf.fit(features, labels)

# the optimal model returned by 'GridSearchCV'
clf = dtcclf.best_estimator_
clf.fit(features, labels)

features_selected_bool = dtcclf.best_estimator_.named_steps['SKB'].get_support()
features_selected_list = np.array(features_list[1:])[features_selected_bool]

# features_list need to be updated before use below!
print 'tree optimization and the result is:'
result = test_classifier(clf, 
                         data_dict, 
                         np.insert(features_selected_list, 0, 'poi'), 
                         folds = 1000)

### Visualize the result
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [0.82713, 0.85667, 0.86893]
bars2 = [0.34533, 0.46265, 0.50912]
bars3 = [0.33100, 0.46450, 0.47450]


# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, 
        bars1, 
        color='#BE6E46', 
        width=barWidth, 
        edgecolor='white', label='Accuracy')
plt.bar(r2, 
        bars2, 
        color='#A1869E', 
        width=barWidth, 
        edgecolor='white', 
        label='Precision')
plt.bar(r3, 
        bars3, 
        color='#59594A', 
        width=barWidth, 
        edgecolor='white', 
        label='Recall')
 
# Add xticks on the middle of the group bars
plt.xlabel('Progression of optimization')
plt.xticks([r + barWidth for r in range(len(bars1))], 
            ['Orginal', 'optimal # of features', 'optimal trees'])
plt.ylabel('A.U.')
plt.ylim(0, 1)
plt.axhline(y = 0.3)
 
# Create legend & Show graphic
plt.legend()
plt.show()

# Visiualize the decision tree
dot_data = tree.export_graphviz(dtcclf.best_estimator_.named_steps['dt'],
                                feature_names=features_selected_list,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                precision=1)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('poi_tree.png')

# The optimal parameters for the decision tree
dtcclf.best_params_

#############################################################################
### Step 6: Dump the classifier, dataset, and features_list to .pkl files ###
#############################################################################
my_dataset = data_dict
dump_classifier_and_data(clf, my_dataset, np.insert(features_selected_list, 0, 'poi'))