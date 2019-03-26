from sklearn.pipeline import make_union

from wookie.pandasconnectors import VectorizerConnector, ExactConnector


def test_loaddata(ix_names, df_left, df_right):
    print(ix_names['ixname'])
    print(df_left.shape[0])
    print(df_right.shape[0])
    assert True


def test_tfidf(df_left, df_right):
    df_X = [df_left, df_right]
    expected_shape = df_left.shape[0] * df_right.shape[0]
    stages = [
        VectorizerConnector(on='name', analyzer='char', pruning=False),
        VectorizerConnector(on='street', analyzer='char', pruning=False),
        ExactConnector(on='duns', pruning=False)

    ]
    scorer = make_union(*stages)
    X_score = scorer.transform(X=df_X)
    assert X_score.shape[0] == expected_shape
    pass
