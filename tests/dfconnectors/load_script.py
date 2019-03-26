import pandas as pd
from sklearn.pipeline import make_union
from wookie.connectors.dataframes import VectorizerConnector, FuzzyConnector

from wookie.preutils import concatixnames

ixname = 'ix'
lsuffix = 'left'
rsuffix = 'right'
ixnameleft, ixnameright, ixnamepairs = concatixnames(
    ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
)
ix_names = dict()
ix_names['ixname'] = ixname
ix_names['ixnameleft'] = ixnameleft
ix_names['ixnameright'] = ixnameright
ix_names['ixnamepairs'] = ixnamepairs
ix_names['lsuffix'] = lsuffix
ix_names['rsuffix'] = rsuffix

df_left = pd.read_csv('/Users/paulogier/81-GithubPackages/wookie/operations/data/left.csv', index_col=0, dtype=str)
df_right = pd.read_csv('/Users/paulogier/81-GithubPackages/wookie/operations/data/right.csv', index_col=0,
                       dtype=str).sample(1000)
df_X = [df_left, df_right]
expected_shape = df_left.shape[0] * df_right.shape[0]
stages = [
    VectorizerConnector(on='name', analyzer='char', vecmodel='cv'),
    VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2)),
    VectorizerConnector(on='name', analyzer='char', vecmodel='tfidf', scoresuffix='chartfidf'),
    VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2)),
    FuzzyConnector(on='name', ratio='simple'),
    FuzzyConnector(on='name', ratio='token')
]
pipe = make_union(*stages)
pipe.fit(X=df_X)
scores = pipe.transform(X=df_X)
alldata = pd.DataFrame(scores, index=stages[0]._getindex(X=df_X), columns=[c.outcol for c in stages])
alldata2 = stages[0].showpairs(X=df_X, y=alldata, use_cols=['name'])
alldata.to_csv('/Users/paulogier/81-GithubPackages/wookie/operations/data/namecomparison3.csv', index=True,
               encoding='utf-8')
