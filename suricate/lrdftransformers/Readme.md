# Left-Right DataFrame Transfomers
## Base
### LrDfTransformerMixin
* Sklearn Transfomer Mixin
* Input: Xlr = [left_dataframe, right_dataframe]
* Transform: Apply a similarity score method to one of the column  of the left and right dataframe
* Output: np.ndarray: of shape(n_samples_left * n_samples_right, 1)

## Cartesian
### Cartesian Join
* function
* returns a side-by-side comparison of the left and right dataframe as a single DataFrame

### Others
* Todo later

## Connector
