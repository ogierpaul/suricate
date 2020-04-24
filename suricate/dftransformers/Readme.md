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

### DfVisualSbs
* Transformer Class
* fit_transform() or transform() have the same effect as a cartesian join above

## Connector
* Base Connector
* Connects two dataframes, one *source* and one *target*
* The similarity matrix is created via a FeatureUnion of the different DfTransformerMixin provided

## Exact
* DfTransformerMixin
* uses vectorized operations
* for each pair of records, returns 1 if it is an exact match, 0 otherwise

## Indexer
* Return the multiindex created from the cartesian join of the let and right index

## Similarities
* DfTransformerMixin
* uses the apply method of pandas dataframe: does not scale very well
* for each pair of records, returns score according to the *comparator* parameter
    * *exact*: 1 or 0
    * *simple*: ratio score of fuzzywuzzy
    * *token*: token_set_ratio score of fuzzywuzzy
    * *vincenty*: Return vincenty distance of two (lat, lng) tuples
    * *contain*: check if one string is a substring of another

## Vectorizer
* DfTransformerMixin
* used for text comparison at (relatively) low computational cost (especially compared to levenshtein distance)
* uses the Sklearn TfIdf Vectorizer or Count Vectorizer to tokenize the text
* The analyzer can be either word or char
* for each pair of tokenized records, returns the cosine similarity score


