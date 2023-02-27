fig= df.plot.hist(column=["PublicationDate"], bins = 5, alpha=0.5, figsize=(8, 8), fontsize=10, range = [1600, 1900]).get_figure()
fig.savefig(SAS.workpath+'test.png', format = 'png', bbox_inches='tight')
SAS.submit('data _null_; declare odsout obj(); obj.image(file:"{}");run;'.format(SAS.workpath+"test.png"))
