# Suricate core modules

### Basic structure
All sub-modules of suricate:
* Base: base classes
* data: data repository (used for testing)
* dbconnectors: special connector for databases (like elastic search)
* explore: help understand the similarity of the datasets by using clusterization
* dftransformers: transformers used for comparing two dataframes
* pipeline: pipelining tools
* preutils: utils for pre-processing the datasets
* sbsdftransformers: transformers used for comparing one dataframe with side-by-side comparison

## Advancement status

|Module|Comments|Features|Tests|Tutorial|Docs|
|---|---|---|---|---|---|
|base|explanation of |Ok|Ok (none)|Ok|Ok|
|data|Done|Ok|Ok|No|Ok|
|dbconnectors|Ok|Ok|Ok|Ok|
|explore|Ok|Ok|Ok|Todo|Ok|
|dftransformers|Ok|ok|ok|Ok|Ok|
|pipeline|Ok|ok|ok|Ok|Ok|
|preutils|to be done later|
|sbstransformers|Ok|ok|ok|Ok|Ok|
|grouping|to be done later|

## Usage Guide
### Purpose
* Two different datasets (source and target) with possible common entities
* Target is to find duplicates : entities in right that are essentially the same as entities in right , +/- typos, missing fields...
* Use Similarity Functions and Machine Learning to identify duplicates automatically

### Deduplication Pipeline
#### 0. Output
* The output of the deduplication pipeline is a vector that identify, for each pair of records from source and target datasets:
    * 1 if the pair is a duplicate
    * 0 if the pair is not a duplicate (no match)

#### 2. Connector to link source and target Datasets
* Will make the link between source and target datasets, whatever the data source (pandas DataFrame, Elastic Search index, Postgre sql database...)
* Is a transformer that transform the source and target datasets into a similarity matrix
* Similarity matrix: Each row corresponds to a pair of (left, right) records, the colums corresponding to different scores
* Will also provide a Side-by-Side view of the data

##### Left dataframe:

|ix|name|
|---|---|
|1|foo|
|2|bar|

##### Right DataFrame:

|ix|name|
|---|---|
|a|foo|
|b|baz|

##### Similarity Matrix:

|Multiindex (ix_source, ix_target)|exact_score|fuzzy_score|
|---|---|---|
|(1,a)|1|1|
|(1,b)|0|0|
|(2,a)|0|0|
|(2,b)|0|0.7|

##### Side-by-side  View
|Multiindex (ix_source, ix_target)|name_source|name_target|
|---|---|---|
|(1,a)|foo|foo|
|(1,b)|foo|baz|
|(2,a)|bar|foo|
|(2,b)|bar|baz|

##### Note
* The example above show a full cartesian join between source and target datasets.
    * Each record in the *source* data is compared with each record in the *target* data
    * it does not offer the best performance for scaling.
* More advanced connectors could offer better scaling thanks to indexing:
    * Elastic Search Connector stores the the *right*  data as an index
    * We profit from Elastic Search Search Capabilities
    * We return the n-best matches from the right_data for each row of the left_dataset)

### 2. Explorer phase using non-supervised techniques
* Cluster the pairs using the similarity matrix
* Ask questions: For a given pair is it a match or is it not a match (not a duplicate)
* Based on this input, identify clusters:
    * Where we there is no match identified
    * where all the pairs are a match
    * Where some pairs are match, some others, and we need to use more features or more advanced machine learning algorithms to classify
* This helps provide a representative sample of the similarity matrix for correct labelling
    
### 3. Optional: Side-by-Side features
* For the records from mixed clusters, apply comparison functions on the side-by-side dataframe to extract more features
* Fuzzy scores, etc...
* Add those scores to the similarity matrix

### 4. Machine-Learning
* Using:
    * the similarity matrix from steps 1 and 3 as features (X)
    * the labelled pairs from step 2 as supervised input (y)
* Train a classifier on the data
* Predict:
    * 1 if the pair is a match
    * 0 if the pair is not a match
 
 #### Example of side-by-side view, similarity matrix, and prediction
 
|Multiindex (ix_source, ix_target)|name_source|name_target|exact_score|fuzzy_score|y_pred|result|
|---|---|---|---|---|---|---|
|(1,a)|foo|foo|1|1|1|Match|
|(1,b)|foo|baz|0|0|0|No|
|(2,a)|bar|foo|0|0|0|No|
|(2,b)|bar|baz|0|0.7|1|Match|



