from pacoetl.utils import neo_graph, pg_engine
import pandas as pd
import numpy as np


# df = pd.read_sql('arp_ypred', con=pg_engine())
sample_id = 'ARP_286025'
q_linked_ids = """
MATCH (a:Arp{arp:$arp})-[:SAME]-(b:Arp)
WHERE a<>b
OPTIONAL MATCH (b)-[:BELONGS]->(g:SupplierGroup)
RETURN  b.arp as arp_target, g.gid as gid, 0 as cost
UNION
MATCH (a:Arp{arp:$arp})-[:HASID|HASNAME]-(d)-[:HASID|HASNAME]-(b:Arp)
WHERE a <> b
OPTIONAL MATCH (b)-[:BELONGS]->(g:SupplierGroup)
RETURN  b.arp as arp_target, g.gid as gid, 1 as cost
"""
r = neo_graph().run(q_linked_ids, parameters={'arp':sample_id}).data()
r = pd.DataFrame(r)
gid = r.dropna(subset='gid').pivot_table(index='gid', values='cost', aggfunc=np.sum).sort_values(ascending=True)
if gid.shape[0] == 0:
    gid = generate_uid()
else:
    gid = gid.iloc[0]
