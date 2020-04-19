# Suricate
A simple and effective framework for finding duplicate entities between datasets.
Based on a modular architecture using Pandas and Scikit-learn base classes (transformer), it is completely customizable and pipelineable.
It also draws heavily on fuzzy matching, using both tf-idf and python-levenshtein (fuzzywuzzy) package.

## Aim: Using machine learning to find duplicate records
### Examples
Duplicate records, or record matching, may occur in different environments:
- Merging two systems of informations (ex: two ERP systems), where you need to identify which supplier companies are the same
- Finding a person between two databases (ex: online survey with email vs windows login)
- ...

The aim is to compare a dataframe (target) with another (right)
- create a similarity matrix between the two set of records
- label the data as 0 --> not a match and 1 --> match
- train a Classifier on the data
- predict

### Basic structure
- Lr (source target) Transformers deal with a variable X=[df_source, df_target] where df_source and df_target have the columns = ['name', 'city', ...]
- Sbs (Side by Side) deal with a variable X=['name_source', 'name_target', 'city_source', 'city_target', ...] (the records are compared side by side)

