|Description|API_Query|Result|
|---|---|---|
|As an authenticated user | | |
|Create a new workspace as an authenticated user and populate it with a node id query|http://localhost:3000/workspace/new?populate=expandNodeId&item_id=1 |ok, we find Keanu Reeves:Person|
|Create a new workspace as an authenticated user and populate it with a search |http://localhost:3000/workspace/new?populate=searchNodes&search_query=Keanu&search_fuzziness=0.0 |ok, we find Keanu Reeves|
|Create a new workspace as an authenticated user and populate it with a cypher (pattern) query|http://localhost:3000/workspace/new?populate=pattern&pattern_query=MATCH%20(n%3APerson)%20RETURN%20n%20LIMIT%205&pattern_dialect=cypher |ok|
|As a guest with guest rights: Can create read-only queries and run existing queries | with all nodes available | |
|Open a blank workspace and search nodes in the config file, set UIWorkspaceSearch for the guest mode = true |http://localhost:3000/guest| Ok, we search for and find, and display KeanuReeves:Person |
|Create a new workspace as an authenticated user and populate it with a node id query|http://localhost:3000/guest?populate=expandNodeId&item_id=1 |ok, we find Keanu Reeves|
|Create a new workspace as an authenticated user and populate it with a search|http://localhost:3000/guest?populate=searchNodes&search_query=Keanu&search_fuzziness=0.0 |ok, we find Keanu Reeves|
|Open a viz as a guest and populate it with a cypher (pattern) query|http://localhost:3000/guest?populate=pattern&pattern_query=MATCH%20(n)%20RETURN%20n%20LIMIT%205&pattern_dialect=cypher |KO: You are not authorized to load the result of this query. You are redirected to an empty visualization|

Testing resume:
* Guest can access the nodes using workspace Search, or through the API with nodeid or search term, it lies not with the guest access rights to Nodes and edges
* Guest cannot run query, even though the rights have been given to him in the user interface, neither in the workspace nor through the API
* The issue potentially lies with the blocking of query rights for the guest mode




