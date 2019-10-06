from suricate.pipeline.pruningpipe import PruningPipe
from suricate.explore import Explorer, KBinsCluster
from suricate.lrdftransformers import LrDfConnector
from tests.lr_pgsqlconnector.notest_setupbase import _score_list, _score_cols, createmultiindex, multiindex21column
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from suricate.data.companies import getXlr
from localpgsqlconnector import pgsqlengine, work_on_ix
import pandas as pd

n_rows = 100
def init_ix():
    Xlr = getXlr(nrows=n_rows)
    connector = LrDfConnector(scorer=None)
    ixc = connector.multiindex21column(on_ix=connector.getindex(Xlr))
    work_on_ix(ixc)
    engine = pgsqlengine()
    query = """
    WITH a AS (SELECT ix, xtrue.y_true FROM tempix
        INNER JOIN xtrue USING(ix)
    ),
    b AS (SELECT ix, ix_left, ix_right FROM xsbs)
    SELECT b.ix_left, b.ix_right, a.y_true 
    FROM b
    INNER JOIN a USING(ix);
    """
    y_true = pd.read_sql(query, con=engine).set_index(['ix_left', 'ix_right'])[['y_true']]
    y_true.to_sql('tempytrue', if_exists='replace', chunksize=2000, method='multi', con=engine)
    engine.dispose()



def comple_workflow():
    print(pd.datetime.now())
    n_cluster = 25
    n_simplequestions = 20
    n_pointedquestions = 40
    engine = pgsqlengine()
    y_true = pd.read_sql('tempytrue', con=engine).set_index(['ix_left', 'ix_right'])['y_true']
    engine.dispose()
    Xlr = getXlr(nrows=n_rows)
    print(pd.datetime.now(), 'data loaded')
    connector = LrDfConnector(
        scorer=Pipeline(
            steps=[
                ('scores', FeatureUnion(_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ]
        )
    )
    explorer = Explorer(
        cluster=KBinsCluster(n_clusters=n_cluster),
        n_simple=n_simplequestions,
        n_hard=n_pointedquestions
    )
    connector.fit(X=Xlr)
    #Xtc is the transformed output from the connector, i.e. the score matrix
    Xtc = connector.transform(X=Xlr)
    print(pd.datetime.now(), 'score ok')
    #ixc is the index corresponding to the score matrix
    ixc = connector.getindex(X=Xlr)
    ix_simple = explorer.ask_simple(X=Xtc, ix=ixc, fit_cluster=True)
    print(pd.datetime.now(), 'length of ix_simple {}'.format(ix_simple.shape[0]))
    sbs_simple = connector.getsbs(X=Xlr, on_ix=ix_simple)
    print('***** SBS SIMPLE ******')
    print(sbs_simple.sample(5))
    print('*****')
    y_simple = y_true.loc[ix_simple]
    ix_hard = explorer.ask_hard(X=Xtc, y=y_simple, ix=ixc)
    print(pd.datetime.now(), 'length of ix_hard {}'.format(ix_hard.shape[0]))
    sbs_hard = connector.getsbs(X=Xlr, on_ix=ix_hard)
    print(sbs_hard.sample(5))
    print('*****')
    y_train = y_true.loc[ix_simple.union(ix_hard)]
    print('length of y_train: {}'.format(y_train.shape[0]))
    explorer.fit(X=pd.DataFrame(data=Xtc, index=ixc), y=y_train)
    print('results of pred:\n', pd.Series(explorer.predict(X=Xtc)).value_counts())
    print('****')
    assert True

if __name__ == '__main__':
    # init_ix()
    comple_workflow()
    # test simple questions
    # test pointed questions
    # test cluster classifier
    # create the sbs scorer
    # create the pipe
    # test the scores
