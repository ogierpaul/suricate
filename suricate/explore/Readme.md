# Explore Module
## Purpose
* Used to explore similarities in the data
* Cluster the pairs based on a similarity matrix
* Can be used then for labelling the data by taking samples for each cluster for manual labelling

## Module Details
### Base
* Base class for the Simple and Hard Questions
* Used to extract samples from each cluster
* Fit with the vector containing the cluster classification of the pairs
* Transform: for each cluster of self.clusters, generate a number of samples

### Simple Questions
From a 1d Vector with the cluster classification of the pairs,
generate for each cluster a number of sample pairs (questions).

This is a simple questions generator because we just use the cluster number to ask the questions,
we do not use any labellized (supervized) data. It is usually the first step in a deduplication process.

### Hard Questions
#### From:
- a 1d Vector with the cluster classification of the pairs
- and with a number of labellized pairs

#### Identify (Fit step):
- Clusters where each of the sample labellized pairs are not matching (nomatch_cluster)
- Clusters where each of the sample labellized pairs are matching (allmatch_cluster)
- Clusters where some of the sample labellized pairs are matching, and some don't (mixedmatch_cluster)

#### Then (Transform step)
- For each of the mixed_match clusters, generate number of questions (Hard questions)

#### Purpose
This is a hard questions generator because we using labellized (supervized) data,
we focus on the similarity cluster where some match and others don't, where the answer is not so obvious: the
frontier between matches and non-matches.

### Cluster Classifier
This Classifier predicts for each cluster if the cluster is :
- no match
- all match
- mixed match

Input data (X) is a (n_pairs, ) pd.Series containing cluster values    
Fit data (y) is a (n_questions, ) pd.Series, with n_questions < n_pairs

### KBins Cluster
* This cluster transformer takes as input a similarity matrix X of size (n_samples, n_features).
* It then sums the score along the n_features axis
* discretize (i.e. Cluster) the scores using the KBinsDiscretizer grom sklearn
* Used to cluster the data according to the similarity score

### Explorer
