# Python or R code based on the Language property.
#
# Note that a few lines of Python or R code are added before your code; for example:
# Python:
#  dm_class_input = ["class_var_1", "class_var_2"]
#  dm_interval_input = ["numeric_var_1", "numeric_var_2"]
# R:
#  dm_class_input <- c("class_var_1", "class_var_2")
#  dm_interval_input <- c("numeric_var_1", "numeric_var_2")
#
# For Python, use the Node Configuration section of the Project Settings to prepend
# any configuration code, which is executed before the above code. During execution,
# this code is automatically prepended to every node that runs Python code.
#
# After running the node, the Python or R code window in the node results displays
# the actual code that was executed. START ENTERING YOUR CODE ON THE NEXT LINE.

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import pickle

X = dm_traindf[dm_input]
Y = dm_traindf.BAD.astype('category')

print(X.head(5))

### creating numerical imputer preprocessor
#numeric_features = dm_traindf.select_dtypes([np.number]).columns
numeric_transformer = Pipeline(
    steps=[("imputer_num", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())]
)
### dropping BAD column
#numeric_features = numeric_features[1:]

### creating categorical imputer and encoding preprocessor
#categorical_features = dm_traindf.select_dtypes([np.object]).columns
categorical_transformer = Pipeline(
    steps=[("imputer_cat", SimpleImputer(strategy="most_frequent")), 
            ("encoder", OneHotEncoder(handle_unknown="ignore"))]
)

#### combining processors
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, dm_interval_input),
        ("cat", categorical_transformer, dm_class_input)
    ]
)

# Fit a sci-kit learn model

dm_model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ('lreg', LogisticRegression(max_iter=1000))
                ]
                )

dm_model.fit(X, Y)

# Score full data: posterior probabilities
dm_inputdf_ = dm_inputdf.drop('BAD',axis =1)

dm_scoreddf_prob = pd.DataFrame(dm_model.predict_proba(dm_inputdf_), columns=dm_predictionvar)

# Score full data: class prediction
dm_scoreddf_class = pd.DataFrame(dm_model.predict(dm_inputdf_), columns=[dm_classtarget_intovar])

# Column merge posterior probabilities and class prediction
dm_scoreddf = pd.concat([dm_scoreddf_prob, dm_scoreddf_class], axis=1)
dm_scoreddf['EM_PROBABILITY'] = np.where(dm_scoreddf["P_BAD1"] > 0.5, dm_scoreddf["P_BAD1"], dm_scoreddf["P_BAD0"])
dm_scoreddf['EM_EVENTPROBABILITY'] = dm_scoreddf["P_BAD1"]
dm_scoreddf['EM_CLASSIFICATION'] = dm_scoreddf['I_BAD']

print('***** 5 rows from dm_scoreddf *****')
print(dm_scoreddf.head(5))
print(dm_input)
print(', '.join(dm_input))


### saving model parameters in report
var_coef = pd.DataFrame({"index": range(0, len(dm_model.named_steps['lreg'].coef_[0])),
               "coef": dm_model.named_steps['lreg'].coef_[0]
              })
     
var_coef.to_csv(dm_nodedir + '/rpt_var_coef.csv', index=False)

# save the model to disk

with open(dm_pklpath, 'wb') as f:
    pickle.dump(dm_model, f)
