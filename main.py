import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


cancerdf = pd.read_csv(r'C:\Users\galmo\training\DS_cancer\CANCER_TABLE.csv')

# count_False = len(cancerdf[cancerdf[" cancer"] == False])
# count_True = len(cancerdf[cancerdf[" cancer"] == True])
# predicted_true = len(cancerdf[(cancerdf["diameter (cm)"] > 7) & (cancerdf[" cancer"] == True)])
# predicted_truefalse = len(cancerdf[(cancerdf["diameter (cm)"] < 7) & (cancerdf[" cancer"] == True)])
# predicted_False = len(cancerdf[(cancerdf["diameter (cm)"] > 7) & (cancerdf[" cancer"] == False)])
# predicted_Falsetrue = len(cancerdf[(cancerdf["diameter (cm)"] < 7) & (cancerdf[" cancer"] == False)])
# print(predicted_Falsetrue, predicted_truefalse)
predicted_true = cancerdf[(cancerdf["diameter (cm)"] > 7) & (cancerdf[" cancer"] == True)]
predicted_truefalse = cancerdf[(cancerdf["diameter (cm)"] < 7) & (cancerdf[" cancer"] == True)]
predicted_False = cancerdf[(cancerdf["diameter (cm)"] > 7) & (cancerdf[" cancer"] == False)]
predicted_Falsetrue = cancerdf[(cancerdf["diameter (cm)"] < 7) & (cancerdf[" cancer"] == False)]
print(confusion_matrix(predicted_true, predicted_truefalse))

data = {'': ['sick', 'not sick'], 'True': [predicted_true, predicted_Falsetrue], 'False': [predicted_truefalse, predicted_False ]}
df = pd.DataFrame(data)
df.set_index('', inplace=True)
print(df)


y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))
