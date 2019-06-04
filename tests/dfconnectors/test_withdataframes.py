from sklearn.pipeline import make_union

from wookie.lrdftransformers import VectorizerConnector, ExactConnector, FuzzyConnector


# from ..data.foo import ix_names
# from ..data.dataframes import df_left, df_right, df_X

def test_loaddata(ix_names, df_left, df_right):
    print(ix_names['ixname'])
    print(df_left.shape[0])
    print(df_right.shape[0])
    assert True


def test_tfidf(df_X):
    expected_shape = df_X[0].shape[0] * df_X[1].shape[0]
    stages = [
        VectorizerConnector(on='name', analyzer='char', pruning=False),
        VectorizerConnector(on='street', analyzer='char', pruning=False),
        ExactConnector(on='duns', pruning=False),
        FuzzyConnector(on='name', ratio='simple')

    ]
    scorer = make_union(*stages)
    X_score = scorer.transform(X=df_X)
    assert X_score.shape[0] == expected_shape
    assert X_score.shape[1] == len(stages)
    pass
