# Source-Target DataFrame Transfomers
- Df (source target) Transformers deal with a variable X=[df_source, df_target] where df_source and df_target have the columns = ['name', 'city', ...]

## Base
### LrDfTransformerMixin
* Sklearn Transfomer Mixin
* Input: Xst = [source_dataframe, target_dataframe]
* Transform: Apply a similarity score method to one of the column  of the source and target dataframe
* Output: np.ndarray: of shape(n_samples_source * n_samples_target, 1)

## Cartesian
### Cartesian Join
* function
* returns a side-by-side comparison of the source and target dataframe as a single DataFrame

### Others
* Todo later

## Connector
* To do later

## Exact
* LrDfTransformerMixin
* uses vectorized operations
* for each pair of records, returns 1 if it is an exact match, 0 otherwise

## Indexer
* To Do later

## Similarities
* LrDfTransformerMixin
* uses the apply method of pandas dataframe: does not scale very well
* for each pair of records, returns score according to the *compfunc* (comparison function)
    * *exact*: 1 or 0
    * *simple*: ratio score of fuzzywuzzy
    * *token*: token_set_ratio score of fuzzywuzzy
    * *vincenty*: Return vincenty distance of two (lat, lng) tuples
    * *contain*: check if one string is a substring of another

## Vectorizer
* LrDfTransformerMixin
* used for text comparison at (relatively) low computational cost (especially compared to levenshtein distance)
* uses the Sklearn TfIdf Vectorizer or Count Vectorizer to tokenize the text
* The analyzer can be either word or char
* for each pair of tokenized records, returns the cosine similarity score


