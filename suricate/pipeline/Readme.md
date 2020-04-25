# Pipelining tools
## 1. Purpose
* The purpose of the pipelines is to assemble several steps in a sequence of transforms until prediction of duplicates
* The main structure highlighted in the Readme of the parent folder is:
    * Connector to connect source and target --> output similarity matrix
    * Explorer to cluster the data --> output clusters
    * Classifier to use similarity and cluster data to classify the pairs
 
## 2. Pipes:

### 2.1 With DataFrames only: (Source-Target dataframes)
#### 2.1.1. PipeDfClf:
#### Steps:
* Source-Target DataFrame Connector
* Classifier

#### 2.1.2. PipeSbsClf:
#### Steps:
* Side-by-Side Comparator functions
* Classifier

### 2.2. With Connectors in general
#### 2.2.1 Pruning Pipe
##### Steps
* connector:DfConnector or EsConnector
* pruningclf: Classifier that prunes (makes a first classification) the data:
    * 0 if it shall be considered as not match
    * 1 if it not sure it is a match or not --> to be analyzed in further steps
    * 2 if it shall be considered as a sure match
* sbsmodel: do side-by-side comparison on the pairs labelled as 1 (mixed matches)
* classifier: classify the mixed matches into:
    * 0: it is not a match
    * 1: it is a match
* Final output:
    * 0 if it is not a match
    * 1 if it is a match
    
#### 2.2.2. PartialClf
* Wrap Around a classifier
* Allows to fit the classifier where the X input data and the y target data are not aligned
* Do a re-index
* Useful when the n_samples of X are not aligned with the saved data y (case of EsConnector, pruning, partial labelling...)
