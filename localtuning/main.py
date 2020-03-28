import localpgsqlconnector
import pandas as pd
engine = localpgsqlconnector.pgsqlengine()
scores = pd.read_sql("SELECT * from xscore WHERE avg_score > 0.5 LIMIT 10;", con=engine)
