## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002,
it had collapsed into bankruptcy due to widespread corporate fraud. In the
resulting Federal investigation, a significant amount of typically confidential
 information entered into the public record, including tens of thousands of
 emails and detailed financial data for top executives.
 In this project, I built machine learning algorithms to identify Enron
 Employees who may have committed fraud (who are called persons of interest,
   POI, in the following) based on the public Enron financial and email dataset.

More specifically, a dataset of persons and features associated with them is
given, and then the goal is to build a classifier algorithm around it that can
predict if individuals should be flagged as POIs or not. This is a dataset of
146 points (before cleaning of outlier points) and 21 features.

## Data exploration
The aggregated Enron email & financial dataset is stored in a dictionary,
where each key-value pair in the dictionary corresponds to one person. The
dictionary key is the person's name, and the value is another dictionary, which
contains the names of all the features and their values for that person. The
 features in the data fall into three major types:

1. financial features (14 in total): e.g salary, loan_advances, bonus, etc
(all units are in US dollars).
2. email features (6 in total): the number of emails received (to_messages),
the number of emails from POI (from_poi_to_this_person), etc (units are
  generally number of emails messages)
3. POI label (1 in total) (1 means that this person is a POI, vice versa)

A brief overview of the dataset is as follows:
1. It includes the information of 146 people.
2. Among them, there 18 POI (person of interest) and 128 non-POI.
3. For each person, there are total 21 features and corresponding values
recorded.

## Outliers investigation
To identify the outliers, I start looking at the salary and bonus of each
person in this dataset. From the scatter plot, I found that a person's salary
is generally proportion to his/her bonus. In addition, red dots are chosen for
POIs and blue dots, non-POIs. However, a non-POI point at the top right corner
appears to be unusually distant from the rest of the data.

![](summary_files/outlier-fig1.png)

Then with the use of list comprehension[1], "total" is identified to be this
distinct observation! In fact, it is likely a spreadsheet quirk since the
spreadsheet added up all the data points automatically as it was generated.

After this removal, I look into the remaining data points again. In this case,
points with salary larger than $1,000,000 or bonus more than $5,000,000 are
considered as outliers. A vertical and a horizontal line are drawn,
respectively, to assist the identification.

![](summary_files/outlier-fig2.png)

Five points fall into this category in which either salary larger than
$1,000,000 or bonus more than $5,000,000. Based on the coloration, three of
these five points are people of interest (red dots). Using a list comprehension,
 the names corresponding to the outliers are:
1. John J Lavorato
2. Kenneth L Lay
3. Timothy N Belden
4. Jeffrey K Skilling
5. Mark A Frevert

It turns out that Kenneth L Lay is the former chairman and CEO, Jeffrey Skilling
 is the former president and COO, and Timothy N Belden is the former head of
 trading in Enron Energy Services. Three of them are definitely people of
 interest. However, the other two people, i.e., John J Lavorato and Mark A
 Frevert seem to be outliers and we may remove them to improve the modeling
 accuracy later.

## Feature selection and optimization
As POIs, they might have particularly strong email connections between each
other. In other words, they send each other emails at a higher rate than people
in the population at large who send emails to POIs. Two new features are
therefore defined to measure the strength of the email connections, which are:

● Percentage of emails from POIs to this person (%) = # emails from POI to this
person / # emails to this person

● Percentage of emails from this person to POIs (%) = # emails from this person
to POI / # emails from this person

They are plotted in the figure below and POIs are in red dots while non-POIs are
in blue dots.

![](summary_files/newfeature-fig1.png)

To figure the importance of each feature which is either financial or email
features, SelectKBest (a univariate feature selection) is used. Note that
'email_address' is not included since it contains text strings, and 'other' is
not used either since this feature is not clearly defined. In addition,
MinMaxScaler is deployed to scale each feature to the range between 0 and 1, and
that score comes from f_classif, which represents ANOVA F-value between label
and features.

![](summary_files/score-fig1.png)

Few observations:
1. The difference between the max and min score is roughly an order of magnitude.
2. The financial features in general have higher ranking than email features.
3. One of the new features "Percentage of emails from this person to POIs (%)"
(fraction_to_poi) is ranked 5th.


## Picking and tuning algorithms
Decision tree and logistic regression are selected to be the algorithms to
build the models. GridCVSearch and Pipeline are used to expedite the
parameters optimization.



## Validation
After the model is built by using the training data set, a separating testing
data set is deployed to make an estimate of the performance of this model (i.e.
  the classifier) on an independent dataset. In addition, the testing data
  serves as check on over-fitting. As to which data is for training or for
  testing, k-fold cross-validation is applied. In k-fold cross-validation, "
   the original sample is randomly partitioned into k equal sized subsamples.
   Of the k subsamples, a single subsample is retained as the validation data
   for testing the model, and the remaining k − 1 subsamples are used as
   training data. The cross-validation process is then repeated k times, with
   each of the k subsamples used exactly once as the validation data. The k
   results can then be averaged to produce a single estimation." [4]
   A set of metrics, including precision, recall and accuracy are used and
   recorded for performance evaluation.

<!---
## Algorithms performance evaluation

## Reflection
--->

References
1. https://discussions.udacity.com/t/encore-des-outliers-2nd-last-part-of-the-outliers-section/31747
1. https://discussions.udacity.com/t/what-are-the-testing-features-when-using-selectkbest/234832/8
2. https://discussions.udacity.com/t/how-to-find-out-the-features-selected-by-selectkbest/45118/4
3. https://discussions.udacity.com/t/how-to-use-pipeline-for-feature-scalling/164178/10
4. https://en.wikipedia.org/wiki/Cross-validation_(statistics)
