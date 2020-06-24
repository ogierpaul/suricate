import datetime
from pacoetl.utils import pg_engine, pg_conn, create_batch, extract_dir
from pacoetl.utils.others import printmessage
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from psycopg2 import sql
from suricate.sbstransformers import SbsApplyComparator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from pacoetl.utils import pg_engine
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

q_eproc = """
SELECT ixp,
       es_score,
       name_fuzzy,
       street_fuzzy,
       city_fuzzy,
       postalcode_fuzzy,
       state_fuzzy,
       countrycode_exact,
       eu_vat_exact,
       concatenatedids,
       cage_exact,
       name_es,
       street_es
FROM (SELECT ixp, ariba_source, arp_target FROM eprocarp.eprocarp_ixp) z
         LEFT JOIN (SELECT ixp,
                           es_score,
                           name_fuzzy,
                           name_token,
                           street_fuzzy,
                           street_token,
                           city_fuzzy,
                           postalcode_fuzzy,
                           state_fuzzy,
                           countrycode_exact,
                           eu_vat_exact,
                           concatenatedids,
                           tax1_contains,
                           cage_exact,
                           duns_exact
                    FROM eprocarp.eprocarp_scores) a USING (ixp)
         LEFT JOIN (SELECT ixp,
                           name_score,
                           pos_score,
                           id_score
                    FROM eprocarp.eprocarp_binned) b USING (ixp)
         LEFT JOIN (SELECT ixp,
                           name_es,
                           street_es
                    FROM eprocarp.eprocarp_namestreet_es) c USING (ixp);

"""

q_arp = """
SELECT ixp,
       es_score,
       name_fuzzy,
       street_fuzzy,
       city_fuzzy,
       postalcode_fuzzy,
       state_fuzzy,
       countrycode_exact,
       eu_vat_exact,
       concatenatedids,
       cage_exact,
       name_es,
       street_es,
       y_true
FROM (SELECT * FROM arpdedupe.arp_ixp) z
LEFT JOIN arpdedupe.arp_scores USING(ixp)
LEFT JOIN (SELECT ixp, name_score_binned AS name_score, pos_score_binned AS pos_score, id_score_binned AS id_score FROM arpdedupe.arp_scores_binned) b USING(ixp)
LEFT JOIN arpdedupe.arp_namestreetes USING (ixp)
LEFT JOIN arpdedupe.arp_ytrue USING(ixp);

"""

q_neo_load_eprocarp_ypred = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.arp_target IS NOT NULL AND row.ariba_source IS NOT NULL
MERGE (a:Arp{arp:row.arp_target})
WITH row, a
MERGE (e:eProc{ariba:row.ariba_source})
WITH row, a, e
MERGE (e)-[r:SAME]->(a)
WITH row, a, e, r
SET r.y_proba = row.y_proba;
"""

def write_y_true():
    df_score = pd.read_sql('SELECT * FROM eprocarp.eprocarp_scores', con=pg_engine()).set_index('ixp')
    df_binned = pd.read_sql('SELECT * FROM eprocarp.eprocarp_binned', con=pg_engine()).set_index('ixp')
    df_sbs = pd.read_sql('SELECT * FROM eprocarp.eprocarp_sbs', con=pg_engine()).set_index('ixp').drop(['es_score'],
                                                                                                       axis=1)
    df_score['es_score'] = df_score['es_score'].astype(float)
    y_pred = pd.Series(index=df_score.index, data=np.zeros(len(df_score.index)), name='y_pred')
    y_pred.loc[(df_score['arp_exact'] == 1)] = 1
    y_pred.loc[(df_score['es_score'] > 70)] = 1
    y_pred.loc[df_binned.loc[df_binned[['name_score', 'pos_score', 'id_score']].sum(axis=1) >= 11].index] = 1
    df_shown = pd.DataFrame(y_pred)
    df_shown = df_shown.join(df_binned, how='left')
    df_shown = df_shown.join(df_sbs, how='left')
    df_shown.to_csv(extract_dir + 'eproc_ytrue.csv', encoding='utf-8', sep='|', index=True)


if __name__ == '__main__':
    df_eproc = pd.read_sql(q_eproc, con=pg_engine()).drop_duplicates(subset=['ixp'], keep='last').set_index(
        'ixp').fillna(0).astype(float)
    df_arp = pd.read_sql(q_arp, con=pg_engine()).set_index('ixp').fillna(0).astype(float)
    df_sbs = pd.read_sql('SELECT * FROM eprocarp.eprocarp_sbs', con=pg_engine()).set_index('ixp')
    y_true = df_arp['y_true'].dropna().astype(int)
    y_true_trim0 = y_true.loc[y_true == 0].sample(5000)
    y_true_trim1 = y_true.loc[y_true == 1]
    y_true = pd.concat([y_true_trim0, y_true_trim1], axis=0, ignore_index=False)
    df_arp.drop(['y_true'], inplace=True, axis=1)
    X = pd.concat([df_arp, df_eproc], axis=0, ignore_index=False, sort=False)
    X_labelled = X.loc[y_true.index]
    y_true = y_true.loc[X_labelled.index]
    k = ShuffleSplit(n_splits=5, test_size=0.33)
    ix_list = []
    for i in k.split(X_labelled):
        idx = X_labelled.iloc[i[0]].index
        ix_list.append(idx)
    gb_params = {'learning_rate': 0.01,
                 'max_depth': 7,
                 'max_features': 'sqrt',
                 'n_estimators': 600}
    pipe1 = Pipeline(steps=[
        ('Scaler', MinMaxScaler()),
        ('PCA', PCA(n_components=7)),
        ('LRC', LogisticRegressionCV(scoring='f1', dual=False, penalty='l2', solver='lbfgs', n_jobs=-1))
    ])
    pipe2 = Pipeline(steps=[
        ('Scaler', StandardScaler()),
        ('PCA', PCA(n_components=7)),
        ('GBC', GradientBoostingClassifier(**gb_params))
    ])
    pipe3 = Pipeline(steps=[
        ('Scaler', StandardScaler()),
        ('PCA', PCA(n_components=6)),
        ('GBC', LogisticRegressionCV(scoring='roc_auc', dual=False, penalty='l2', solver='lbfgs', n_jobs=-1))
    ])
    pipe4 = Pipeline(steps=[
        ('Scaler', StandardScaler()),
        ('PCA', PCA(n_components=6)),
        ('GBC', GradientBoostingClassifier(**gb_params))
    ])
    results = pd.DataFrame(index=df_eproc.index)
    d_control = pd.DataFrame(index=X_labelled.index)
    printmessage('start')
    for (i, p, n) in zip(range(4), [pipe1, pipe2, pipe3, pipe4], ['y_pred1', 'y_pred2', 'y_pred3', 'y_pred4']):
        printmessage(n)
        p.fit(X_labelled.loc[ix_list[i]], y_true.loc[ix_list[i]])
        printmessage('fitted')
        y_control = pd.Series(index=X_labelled.index, data=p.predict(X_labelled))
        printmessage('precision score :{}\nrecallscore:{}\nf1score:{}'.format(
            precision_score(y_true, y_control),
            recall_score(y_true, y_control),
            f1_score(y_true, y_control)
        )
        )
        y = p.predict_proba(df_eproc)[:, 1]
        y_control = pd.Series(index=X_labelled.index, data=p.predict_proba(X_labelled)[:, 1], name=n)
        printmessage('predicted\n')
        y = pd.Series(index=df_eproc.index, data=y, name=n)
        results[n] = y
        d_control[n] = y_control
    d_control['y_proba_mean'] = d_control.mean(axis=1)
    d_control['y_proba_max'] = d_control.max(axis=1)
    printmessage('\nfor group score')
    for c in ['y_proba_mean', 'y_proba_max']:
        printmessage('\n {}'.format(c))
        y_control = (d_control[c]>0.5)
        printmessage('precision score :{}\nrecallscore:{}\nf1score:{}'.format(
            precision_score(y_true, y_control),
            recall_score(y_true, y_control),
            f1_score(y_true, y_control)
        )
    )
    results['y_proba'] = results.mean(axis=1)
    results.loc[(df_sbs['arp_source'] == df_sbs['arp_target']), 'y_proba'] = 0.51
    results['y_pred'] = (results.max(axis=1) >= 0.5).astype(int)
    df_sbs['y_proba'] = results['y_proba']
    df_sbs['y_pred'] = results['y_pred']
    df_sbs[df_sbs['y_pred'] == 1].to_sql('eprocarp_ypred', schema='eprocarp', con=pg_engine())
    # df_sbs.locto_csv(extract_dir+'eprocarp_ypred.csv', index=True, sep='|', encoding=False)

