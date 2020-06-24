from pacoetl.utils.clean import clean_inputs, drop_na_dupes, sanitize_csv, sanitize_js, convertnone, zeropadding, castcols, select_rename_order_cols
from pacoetl.utils.pg import pg_copy_from, pg_insert, pg_conn, pg_engine
from pacoetl.utils.esutils import es_index, es_create, es_create_load, es_client
from pacoetl.utils.neo import neo_graph, neo_bulk_import
from pacoetl.utils.others import create_batch, deduperels

extract_dir = '/Users/paulogier/81-GithubPackages/suricate/project/data_dir/extract_dir/'
staging_dir = '/Users/paulogier/81-GithubPackages/suricate/project/data_dir/staging/'
neo4j_dir = '/Users/paulogier/84-neo4j_home/neo4j-community-3.5.15/import/'