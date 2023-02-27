# A score function of the following form should be provided:
# def score_record(var_1, var_2, var_3, var_4):
#   "Output: outvar_1, outvar_2"
#   <code line 1>
#   <code line 2 and so on>
#   return outvar_1, outvar_2
#
# Note that the pickle file saved at dm_pklpath in the Training Code editor can be opened with the following code:
# open(settings.pickle_path + dm_pklname)

import pickle
import numpy as np
import pandas as pd

# Load pickle file objects
with open(settings.pickle_path + dm_pklname, 'rb') as f:
    model = pickle.load(f)

def score_method(LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC):
    "Output: P_BAD0, P_BAD1, I_BAD, EM_EVENTPROBABILITY, EM_CLASSIFICATION, EM_PROBABILITY"
    
    # Create single row dataframe
    record = pd.DataFrame([[LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],\
             columns=['LOAN', 'MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'])

    # Score data passed to this method
    rec_pred_prob = model.predict_proba(record)
    rec_pred = model.predict(record)

    EM_PROBABILITY = rec_pred_prob[0][1] if rec_pred_prob[0][1] >= 0.5 else rec_pred_prob[0][0]

    return float(rec_pred_prob[0][0]), float(rec_pred_prob[0][1]), float(rec_pred[0]), float(rec_pred_prob[0][1]), float(rec_pred[0]), float(EM_PROBABILITY)