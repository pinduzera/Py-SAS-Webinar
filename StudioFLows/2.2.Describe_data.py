
df2 = df.describe()
df2.reset_index(inplace=True)
df2 = df2.rename(columns = {'index':'Index_name'})

SAS.df2sd(df2, _output2)