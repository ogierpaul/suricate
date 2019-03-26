from wookie.comparators import FuzzyWuzzySbsComparator


def test_cclr(df_circus, circus_sbs):
    expected_shape = df_circus[0].shape[0] * df_circus[1].shape[0]
    assert circus_sbs.shape[0] == expected_shape
    print(circus_sbs)
    comp = FuzzyWuzzySbsComparator(on_left='name_left', on_right='name_right', comparator='fuzzy')
    X_score = comp.transform(circus_sbs)
    assert X_score.shape[0] == expected_shape
    circus_sbs['score'] = X_score
    print(circus_sbs.sort_values(by='score', ascending=False))
