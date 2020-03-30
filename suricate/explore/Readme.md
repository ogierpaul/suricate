# Explore Module
## Purpose
* Used to explore similarities in the data
* Cluster the pairs based on a similarity matrix
* Can be used then for labelling the data by taking samples for each cluster for manual labelling

## Module Details
### Questions
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

### Explorer

### Cluster Classifier


### KBins Cluster

