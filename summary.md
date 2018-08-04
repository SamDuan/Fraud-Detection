## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002,
it had collapsed into bankruptcy due to widespread corporate fraud. In the
resulting Federal investigation, a significant amount of typically confidential
 information entered into the public record, including tens of thousands of
 emails and detailed financial data for top executives.
 In this project, I built machine learning algorithms to identify Enron
 Employees who may have committed fraud (who are called persons of interest,
   POI, in the following) based on the public Enron financial and email dataset.

## Data exploration
The aggregated Enron email & financial dataset is stored in a dictionary,
where each key-value pair in the dictionary corresponds to one person. The
dictionary key is the person's name, and the value is another dictionary, which
contains the names of all the features and their values for that person. The
 features in the data fall into three major types:

1. financial features (14 in total): ['salary', 'deferral_payments',
'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
(all units are in US dollars)

2. email features (6 in total): ['to_messages', 'email_address',
'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
'shared_receipt_with_poi'] (units   are generally number of emails messages;
  notable exception is ‘email_address’, which is a text string)

3. POI label (1 in total): [‘poi’] (boolean, represented as integer)

Dictionary is stored as a pickle file, which is the format of the data source
for this project. A brief overview of the dataset is as follows:
1. It includes the information of 146 people.
2. Among them, there 18 POI (person of interest) and 128 non-POI.
3. For each person, there are 21 features and corresponding values recorded.

## Outliers investigation
To identify the outliers, we start looking at the salary and bonus of each
person in this dataset. From the scatter plot, we found that a person's salary
is generally proportion to his/her bonus. However, a point at the top right
corner appears to be unusually distant from the rest of the data.

![](summary_files/outlier-fig1.png)

Then with the use of list comprehension, "total" is identified to be this
distinct observation! In fact, it is likely a spreadsheet quirk since the
spreadsheet added up all the data points automatically as it was generated.

After this removal, we look into the remaining data points again. In this case,
points with salary larger than $1,000,000 or bonus more than $5,000,000 are
considered as outliers. A vertical and a horizontal line are drawn,
respectively, to assist the identification.

![](summary_files/outlier-fig2.png)

Five points fall into this category in which either salary larger than
$1,000,000 or bonus more than $5,000,000. In fact, it is quite possible that
these five points are people of interest. Using a list comprehension, the names
corresponding to the outliers are:
1. John J Lavorato
2. Kenneth L Lay
3. Timothy N Belden
4. Jeffrey K Skilling
5. Mark A Frevert

It turns out that Kenneth L Lay is the chairman and CEO, and Jeffrey Skilling is
 the president and COO. Both are definitely people of interest, so they cannot
 be removed. However, the other three people, i.e., John J Lavorato,
 Timothy N Belden and Mark A Frevert, are removed from the dataset.

## Feature selection and optimization

## Picking and tuning algorithms

## Validation

## Algorithms performance evaluation

## Reflection
