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

1. financial features: ['salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees']
(all units are in US dollars)

2. email features: ['to_messages', 'email_address', 'from_poi_to_this_person',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units
  are generally number of emails messages; notable exception is ‘email_address’,
   which is a text string)

3. POI label: [‘poi’] (boolean, represented as integer)

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

Then with the use of list comprehension, "total" is identified to be this
distinct observation! In fact, it is likely a spreadsheet quirk since the
spreadsheet added up all the data points automatically as it was generated.

After this removal, we look into the remaining data points again. At this time,
points with salary larger than 1e6 or bonus more than 5e6 are considered as
outliers. A vertical and a horizontal line are drawn, respectively, to assist
us to identify them. 

## Feature selection and optimization

## Picking and tuning algorithms

## Validation

## Algorithms performance evaluation

## Reflection
