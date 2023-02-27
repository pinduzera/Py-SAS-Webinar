# %%
# !pip install swat

# %%
import swat
import getpass
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# %% [markdown]
# # Conexão com servidor

# %%
# slower http/REST connection method
conn = swat.CAS("https://my-viya.com/cas-shared-default-http", 443, "username", getpass.getpass())
# print(conn)

# fast binary connection method

#conn = swat.CAS("controller.sas-cas-server-default.viya4.svc.cluster.local", 5570, 
#                username = "viyademo02", 
#                password = getpass.getpass(), 
#                ssl_ca_list="/home/jovyan/pythondata/certs/trustedcerts.pem")
print(conn)

# %%
# https://go.documentation.sas.com/doc/en/pgmsascdc/v_035/allprodsactions/actionSetsByName.htm

conn.loadactionset("sampling")
conn.loadactionset("decisionTree")
conn.loadactionset("autotune")

# %% [markdown]
# # Verificar Tabelas

# %%
# Tabelas em memoria
conn.tableinfo(caslib="CASUSER")

# %%
conn.upload(data='hmeq.csv', casout={"name":"hmeq","caslib":"casuser", "replace": True})
conn.tableinfo(caslib="CASUSER")

# %% [markdown]
# # Análise da Tabela

# %%
tbl = conn.CASTable("hmeq", caslib="CASUSER")

# %%
tbl.head()

# %%
tbl.describe()

# %%
# 0 = trainamento, 1 = teste
conn.sampling.stratified(
    table={"name":"hmeq", "groupBy":"BAD"},
    output={"casOut":{"name":"hmeq_part", "replace":1}, "copyVars":"ALL"},
    samppct=70,
    partind=True
)

# %%
tbl_part = conn.CASTable("hmeq_part", caslib="CASUSER")
tbl_part.head()

# %% [markdown]
# # Definição de Variáveis

# %% [markdown]
# ## Imputação de Valores

# %%
# Imputacao de Valores
db_var_imp = conn.datapreprocess.impute(table="hmeq_part",
                                        methodnominal="mode", 
                                        methodinterval ="median",
                                        casout={"name":"HMEQ_TRATADA","caslib":"CASUSER", "replace":1},
                                        outvarsnameprefix='')

db_tratado = conn.CASTable("HMEQ_TRATADA")
conn.table.promote(db_tratado)

db_tratado.head()

# %%
# Separacao de Colunas
columns_info = conn.columninfo(table=db_tratado).ColumnInfo

target = "BAD"
columns_info


# %%

columns_char = list(columns_info["Column"][columns_info["Type"]=="varchar"])
columns_double = list( columns_info["Column"][ columns_info["Type"]=="double" ])
columns_double.remove("BAD")
columns_double.remove("_PartInd_")

print(columns_char)
print(columns_double)

# %% [markdown]
# # Criação do Modelo

# %% [markdown]
# ## Random Forest

# %%
# Treinamento e Scoragem - Random Forest

resultrf = conn.autotune.tuneForest(
    # Treina e salva o codigo de treinamento na tabela rf_train.
    trainOptions={
         "table"   : {"name":"hmeq_part", "where": "_PartInd_=0"},
         "inputs"  : columns_double+columns_char,
         "target"  : target,
         "nominals" : columns_char+[target],
         "casout"  : {"name":"rf_train"},
        "saveState" : {"name" : "rf_astore", "caslib": "Public"} # astore
     },
    tunerOptions={
         "maxIters": 5,
         "maxTime": 60,
         "searchMethod": "GA",
         "objective": "KS",
         "userDefinedPartition": True,
         "targetEvent" : "1"
     },
    # Utiliza o modelo criado e otimizado para scoragem da base particionada
    scoreOptions = {
        "table": { "name":"hmeq_part", "where": "_PartInd_=1" },
        "modeltable": {"name":"rf_train"},
        "casout": {"name":"rf_score", "replace":1},
        "copyvars":["BAD"]
    }
)

# Scoragem - Random Forest
rf_score = conn.CASTable("rf_score") 
rf_score.head()

# %% [markdown]
# ## Gradient Boosting

# %%
# Treinamento e Scoragem - Gradient Boosting
resultgb = conn.autotune.tuneGradientBoostTree(
    trainOptions = {
        "table"   : {"name":"hmeq_part", "where": "_PartInd_=0"},
        "inputs"  : columns_double+columns_char,
        "target"  : target,
        "nominal" : columns_char+[target],
        "casout"  : {"name":"gb_train"},
        "saveState" : {"name" : "gb_astore", "caslib": "Public"}
    },
    tunerOptions={
         "maxIters": 5,
         "maxTime": 60,
         "searchMethod": "GA",
         "objective": "KS",
         "userDefinedPartition": True,
         "targetEvent" : "1"
    },
    scoreOptions= {
        "table" : {"name":"hmeq_part", "where": "_PartInd_=1"},
        "modeltable": {"name":"gb_train"},
        "casout":{"name":"gb_score", "replace":1}, 
        "copyvars":["BAD"]
   }
)

gb_score = conn.CASTable("gb_score")
gb_score.head()

# %%
resultnn =conn.autotune.tuneneuralnet(
    trainOptions = {
        "table"   : {"name":"hmeq_part", "where": "_PartInd_=0"},
        "inputs"  : columns_double+columns_char,
        "target"  : target,
        "nominal" : columns_char+[target],
        "casout"  : {"name":"nn_train"}
    },
    tunerOptions={
         "maxIters": 5,
         "maxTime": 60,
         "searchMethod": "GA",
         "objective": "KS",
         "userDefinedPartition": True,
         "targetEvent" : "1"
    },
    scoreOptions= {
        "table" : {"name":"hmeq_part", "where": "_PartInd_=1"},
        "modeltable": {"name":"nn_train"},
        "casout":{"name":"nn_score", "replace":1}, 
        "copyvars":["BAD"]
   }
)

nn_score = conn.CASTable("gb_score")
nn_score.head()

# %% [markdown]
# # Informações dos Modelos

# %%
resultgb

# %%
conn.tableinfo(caslib="CASUSER")

# %%
conn.tableinfo(caslib="Public")

# %% [markdown]
# # Gráfico de Assessment

# %%
metric = "KS"

# %%
resultgb["ROCInfo"][resultgb["ROCInfo"][metric] == max(resultgb["ROCInfo"][metric])]


# %%
resultrf["ROCInfo"][resultrf["ROCInfo"][metric] == max(resultrf["ROCInfo"][metric])]

# %%
resultnn["ROCInfo"][resultnn["ROCInfo"][metric] == max(resultnn["ROCInfo"][metric])]

# %%
plt.plot(resultrf["ROCInfo"]["FPR"], resultrf["ROCInfo"]["Sensitivity"])
plt.plot(resultgb["ROCInfo"]["FPR"], resultgb["ROCInfo"]["Sensitivity"])
plt.plot(resultnn["ROCInfo"]["FPR"], resultnn["ROCInfo"]["Sensitivity"])
plt.plot([0,1],linestyle="dashed", color = 'black')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend(["RandomForest :"+ str(round(resultrf["ROCInfo"]["C"][0], 3)),
            "GradientBoost :"+ str(round(resultgb["ROCInfo"]["C"][0], 3)),
            "NeuralNet :"+ str(round(resultnn["ROCInfo"]["C"][0], 3))])


# %%
from sklearn.metrics import classification_report

# %%
gbdf = gb_score.to_frame()
rfdf = rf_score.to_frame()

# %%
print("## GBT metrics")
print(classification_report(y_true = gbdf["BAD"].astype(int), 
                            y_pred = gbdf["I_BAD"].astype(int), 
                            target_names = ["0", "1"]))

# %%
print("## RF metrics")
print(classification_report(y_true = rfdf["BAD"].astype(int), 
                            y_pred = rfdf["I_BAD"].astype(int), 
                            target_names = ["0", "1"]))

# %% [markdown]
# ## Salvando, registrando e publicando Modelos

# %%
## download the model
conn.loadactionset('aStore')

store=conn.download(rstore= {"name": "gb_astore", "caslib": "Public"})

with open('savelocal.sasast','wb') as file:
   file.write(store['blob'])

# %%
from sasctl import Session
from sasctl.tasks import register_model, publish_model
import getpass

# %%
# Establish a session with Viya
s= Session("https://viyatogo-singlenode", 
            username  = "username",
            password = getpass.getpass()
       )
            
print(s)

# %%
result = conn.astore.describe(rstore= dict(name = "gb_astore", caslib= "Public" ), epcode=False)
var_list = [print(v) for v in result.InputVariables.itertuples()]



# %%
astore = conn.CASTable('gb_astore', caslib = "public")
model = register_model(astore, 'gb_swat', 'WebinarBrHmeq') #force = True to create the project

# %%
# Publicar o modelo para scoragem em tempo real
module = publish_model(model, 'maslocal')


# %%
first_rows = tbl.head(10)

# %%
# Enviando uma linha para MAS e rebendo a predição.
result = module.score(first_rows.iloc[8])
print(result)

# %%
s.delete()

# %%
conn.terminate()

# %% [markdown]
# ## Modelo Python

# %%
import pandas as pd
from sasctl import Session, register_model, publish_model
from sklearn.linear_model import LogisticRegression


# %%
# Load the Iris data set and split into features and target.
df = pd.read_csv('https://support.sas.com/documentation/onlinedoc/viya/exampledatasets/iris.csv')
df.columns = df.columns.str.replace(' ', '_')
df.drop(["Index"], axis=1, inplace = True)

X = df.drop('Species', axis=1)
y = df.Species.astype('category')

# %%
df.head(10)

# %%
# Fit a sci-kit learn model
model = LogisticRegression()

# %%
model.fit(X, y)

# %%

# Establish a session with Viya
Session("https://my-viya-server.com", 
            username  = "username", 
            password = getpass.getpass())


# %%
model_name = 'IrisLogRegression'

# Register the model in Model Manager
register_model(model,
                   model_name,
                   input=X,         # Use X to determine model inputs
                   project='IrisProject',  # Register in "Iris" project
                   force=True)      # Create project if it doesn't exist

# %%
# Publish the model to the real-time scoring engine
module = publish_model(model_name, 'maslocal')

# %%
# Select the first row of training data
x = X.iloc[100, :] #

# Call the published module and score the record
result = module.predict(x)
print(result)

# %%

# deletando modulo (publicação)

from sasctl import delete
delete("/microanalyticScore/modules/irislogregression_338e183c7aca42")

# %%



