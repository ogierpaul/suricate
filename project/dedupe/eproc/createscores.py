import datetime
import numpy as np
from pacoetl.utils import pg_engine, pg_conn, create_batch
import pandas as pd
from sklearn.pipeline import FeatureUnion
from psycopg2 import sql
from suricate.sbstransformers import SbsApplyComparator
from suricate.dftransformers import VectorizerConnector, createmultiindex


def tfidf_scores(ixp, df_source, df_target):
    tfidf_score_list = [
        ('name_tfidf_char', VectorizerConnector(on='name', ngram_range=(2, 3), analyzer='char')),
        ('name_tfidf_word', VectorizerConnector(on='name', ngram_range=(1, 2), analyzer='word')),
        ('street_tfidf_char', VectorizerConnector(on='street', ngram_range=(2, 3), analyzer='char')),
        ('street_tfidf_word', VectorizerConnector(on='street', ngram_range=(1, 2), analyzer='word'))
    ]
    df_source.index.name = 'ix'
    df_target.index.name = 'ix'
    ix_source = ixp.apply(lambda r: r.split('-')[0])
    ix_target = ixp.apply(lambda r: r.split('-')[1])
    df_source = df_source.loc[ix_source]
    df_target = df_source.loc[ix_target]
    Xlr = [df_source, df_target]
    pipe = FeatureUnion(transformer_list=tfidf_score_list)
    Xst = pipe.fit_transform(X=Xlr)
    Xst = pd.DataFrame(index=createmultiindex(X=Xlr), columns=[c[0] for c in tfidf_score_list], data=Xst)
    Xst.reset_index(drop=False, inplace=True)
    return Xst


def scores_further(df):
    sbs_score_list = [
        ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
        ('name_token', SbsApplyComparator(on='name', comparator='token')),
        ('street_fuzzy', SbsApplyComparator(on='street', comparator='simple')),
        ('street_token', SbsApplyComparator(on='street', comparator='token')),
        ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
        ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple')),
        ('state_fuzzy', SbsApplyComparator(on='state', comparator='simple')),
        ('countrycode_exact', SbsApplyComparator(on='countrycode', comparator='exact')),
        ('eu_vat_exact', SbsApplyComparator(on='eu_vat', comparator='exact')),
        ('concatenatedids', SbsApplyComparator(on='eu_vat', comparator='token')),
        ('tax1_contains', SbsApplyComparator(on='tax1', comparator='contains')),
        ('cage_exact', SbsApplyComparator(on='cage', comparator='exact')),
        ('duns_exact', SbsApplyComparator(on='duns', comparator='exact')),
        ('arp_exact', SbsApplyComparator(on='arp', comparator='exact')),
    ]
    scorer_sbs = FeatureUnion(transformer_list=sbs_score_list)
    out = scorer_sbs.fit_transform(X=df)
    out = pd.DataFrame(data=out, index=df.index, columns=[c[0] for c in sbs_score_list])
    return out


def createscores():
    df = pd.read_sql("""
    SELECT
        *
    FROM eprocarp.eprocarp_sbs;
    """, con=pg_engine()).set_index(['ixp'])
    print(datetime.datetime.now(), ' | data loaded with {}'.format(df.shape[0]))
    dfb = create_batch(df, batch_size=10000)
    con = pg_conn()
    cur = con.cursor()
    for i, d in enumerate(dfb):
        print(datetime.datetime.now(), ' | Start batch {} of {}'.format(i + 1, len(dfb)))
        Xsb = scores_further(d)
        Xsb['es_score'] = df['es_score']
        print(datetime.datetime.now(), ' | SBS scores calculated')
        Xsb.to_sql('eprocarp_scores', if_exists='append', con=pg_engine())
        print(datetime.datetime.now(), ' | loaded in PG\n')
    cur.close()


def discretizescores():
    df = pd.read_sql('SELECT * FROM eprocarp_scores', con=pg_engine()).set_index('ixp')
    from sklearn.preprocessing import KBinsDiscretizer

    n_bins = 5
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    namecols = ['name_fuzzy', 'name_token']
    poscols = ['street_fuzzy', 'street_token', 'city_fuzzy', 'postalcode_fuzzy', 'state_fuzzy', 'countrycode_exact']
    idcols = ['eu_vat_exact', 'concatenatedids', 'tax1_contains', 'cage_exact',
              'duns_exact', 'arp_exact']
    df['name_score'] = df[namecols].mean(axis=1)
    df['pos_score'] = df[poscols].mean(axis=1)
    df['id_score'] = df[idcols].mean(axis=1)
    usecols = ['es_score', 'name_score', 'pos_score', 'id_score']
    df = df[usecols].fillna(0)
    df2 = kb.fit_transform(X=df)
    df2 = pd.DataFrame(df2, columns=usecols, index=df.index)
    df2.sort_values(by='es_score', ascending=False, inplace=True)
    for c in usecols:
        df2[c] = df2[c] + 1
    df2.to_sql('eprocarp_binned', con=pg_engine(), schema='eprocarp', if_exists='replace')


if __name__ == '__main__':
    df_ixp = pd.read_sql('SELECT ixp, arp_source, arp_target FROM arpdedupe.arp_ixp WHERE es_score IS NOT NULL and es_score>25 ;', con=pg_engine()).set_index('ixp')
    df_ixp.columns = ['ix_source', 'ix_target']
    ixnamepairs = df_ixp.columns
    df_source = pd.read_sql('SELECT arp, name, street FROM paco.eproc;', con=pg_engine()).set_index('arp')
    df_target = pd.read_sql('SELECT arp, name, street FROM paco.arp;', con=pg_engine()).set_index('arp')
    df_source.index.name = 'ix'
    df_target.index.name = 'ix'
    dfb = create_batch(df_ixp, batch_size=5*10**3)
    tfidf_score_list = [
        ('name_tfidf_char', VectorizerConnector(on='name', ngram_range=(2, 3), analyzer='char')),
        ('name_tfidf_word', VectorizerConnector(on='name', ngram_range=(1, 2), analyzer='word')),
        ('street_tfidf_char', VectorizerConnector(on='street', ngram_range=(2, 3), analyzer='char')),
        ('street_tfidf_word', VectorizerConnector(on='street', ngram_range=(1, 2), analyzer='word'))
    ]
    for (i, d) in enumerate(dfb):
        print(datetime.datetime.now(), i, len(dfb))
        ix_source = np.intersect1d(d['ix_source'].drop_duplicates(), df_source.index)
        ix_target = np.intersect1d(d['ix_target'].drop_duplicates(), df_target.index)
        ix_common = d.loc[(d['ix_source'].isin(ix_source)) & (d['ix_target'].isin(ix_target))]
        ix_common = ix_common.set_index(list(ixnamepairs)).index
        Xlr = [df_source.loc[ix_source], df_target.loc[ix_target]]
        pipe = FeatureUnion(transformer_list=tfidf_score_list)
        Xst = pipe.fit_transform(X=Xlr)
        Xst = pd.DataFrame(index=createmultiindex(X=Xlr), columns=[c[0] for c in tfidf_score_list], data=Xst)
        Xst = Xst.loc[ix_common]
        Xst.reset_index(drop=False, inplace=True)
        Xst['ixp'] = Xst.apply(lambda r:'-'.join(r[['ix_source', 'ix_target']]), axis=1)
        Xst.to_sql('arp_tfidf', con=pg_engine(), schema='arpdedupe', if_exists='append')




