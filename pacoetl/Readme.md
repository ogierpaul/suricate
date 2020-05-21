# Paco Project
Paco Project aims to create the heavy lifting of the data engineering behind the suricate project.

## Data Flow Description
### Extract - Sources of data
- Arp
- Ariba
- Dnb
- Stored in a path called extract_path
- Has a time stamp extract_ts
- Extract area

### Transform
#### Data Cleansing
- Select the cols
- Rename the cols
- Format the data to text, integer, etc..
- Fill the leading zeroes
- Remove the null or n.a. values
- Remove duplicate
- Apply filters
- Prevent JS injections

#### Data Modelling & Enrichment
- Include Geocoding (OSM)
- Separate the entities (Neo4j)
- Add common matching fields (Suricate)

#### Output
- One cleaned csv file in a staging area per output

### Load
#### Target
- Load into Postgresql
- Load into Neo4j
- Load into Elastic Search

#### Method
- Use the COPY method for speed
- OR use the INSERT method
- Prevent SQL injections attack
- Include a timestamp for last time row was extracted and ingested --> max is updated_ts
- Include a timestamp for the time the row was created 

#### 
- Include