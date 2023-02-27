import numpy as np

pub = df['PublicationPlace']
london = pub.str.contains('London')
oxford = pub.str.contains('Oxford')

df['PublicationPlace'] = np.where(london, 'London',
                                      np.where(oxford, 'Oxford',
                                               pub.str.replace('-', ' ')))
df['PublicationPlace']
SAS.df2sd(df, _output1)