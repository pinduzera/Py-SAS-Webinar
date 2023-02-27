# Using PROC PYTHON Callback Methods
# https://go.documentation.sas.com/doc/en/pgmsascdc/v_017/proc/p1m1pc8yl1crtkn165q2d4njnip1.htm#n1x71i41z1ewqsn19j6k9jxoi5fa

# submit SAS code
SAS.submit("data test; x='python'; y=2; run;")

# call a SAS function
py_var = SAS.sasfnc("UPCASE", "HeLlO wOrLd")
print(py_var)

# get a SAS macro variable
py_var=SAS.symget("sysvlong")
print(py_var)

# put a SAS macro variable
SAS.symput("setfrompy", "hello world")

# SAS data set to Pandas DataFrame
df=SAS.sd2df('sashelp.class(keep=name sex age)')

# Pandas DataFrame to SAS data set
SAS.df2sd(df, 'work.backToSAS')
