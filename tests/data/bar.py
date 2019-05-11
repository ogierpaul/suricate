from ..data.foo import ix_names as _ix_names, df_left as _df_left, df_right as _df_right, df_sbs as _df_sbs

ix_names = _ix_names()
df_left = _df_left(ix_names=ix_names)
df_right = _df_right(ix_names=ix_names)
df_sbs = _df_sbs(ix_names=ix_names)
df_X = [df_left, df_right]
