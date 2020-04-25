# Side-by-Side comparators
* Sbs (Side by Side) deal with a variable X=['name_source', 'name_target', 'city_source', 'city_target', ...] (the records are compared side by side)
* Perform comparison on two different columns of a same table (like in side-by-side view)
* Uses different comparison functions: ['exact', 'simple', 'token', 'contains', 'vincenty' ]
* Example of side-by-side data:

##### Side-by-side  View
|Multiindex (ix_source, ix_target)|name_source|name_target|
|---|---|---|
|(1,a)|foo|foo|
|(1,b)|foo|baz|
|(2,a)|bar|foo|
|(2,b)|bar|baz|

