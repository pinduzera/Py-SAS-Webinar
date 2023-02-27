df=SAS.sd2df(_input1)
df['PublicationDate'] = df['PublicationDate'].str.extract(r'^(\d{4})', expand=False)
df['PublicationDate'] = df['PublicationDate'].astype(int)
