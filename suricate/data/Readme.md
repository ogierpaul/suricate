# Sample datasets
This module lets you access sample datasets to test your pipelines.
We have three datasets:
* foo
* circus
* companies

## Loading the datasets
Each dataset can be accessed by three methods:
* getsource(), gettarget(): returns source and target datasets as Pandas DataFrame
* getXst(): returns a list of length 2 containing the source and target datasets
* getXsbs(): returns a Pandas DataFrame containing the cartesian product of the source and target datasets (not available for companies because it would be too huge)
* getytrue(): returns a Pandas Series containing the matching information for the datasets

## Dataset description
* foo: two simple datasets of shape (3, 1) without nulls
* circus: two identicatal datasets of shape (6, 1) with closely matching records with more complicated input (null, casing, spaces)
* companies: two different datasets of length (3177, 6) and (1444, 6) with realistic mock-up data (company name, street, city, postalcode, id number...)


