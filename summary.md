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

## Feature selection and optimization

## Picking and tuning algorithms

## Validation

## Algorithms performance evaluation

## Reflection
