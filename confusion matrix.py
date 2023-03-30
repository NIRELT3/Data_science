from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def tpr(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def recall(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def fpr(false_poitive, true_negative):
    return false_poitive / (false_poitive + true_negative)


def precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)


def fscore(precision, recall):
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1


def accuracy(true_positive, false_negative, true_negative, false_positive):
    return (true_positive + true_negative) / (true_positive + false_negative + true_negative + false_positive)


df = pd.read_csv(r'C:\Users\galmo\training\DS_cancer\CANCER_TABLE.csv')

# filter the data frame to include only rows where diameter > 7
df_filtered = df[df['diameter (cm)'] > 7]

y_true = df_filtered[' cancer'].astype(int)

y_pred = (df_filtered['diameter (cm)'] > 7).astype(int)

# calculate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
# print the confusion matrix
print(conf_matrix)
print(TN, FP, FN, TP, TPR, FPR)

df_filtered = df[df['diameter (cm)'] > 7]

true_positives = len(df_filtered[(df_filtered[' cancer'] == True)])
false_negatives = len(df_filtered[(df_filtered[' cancer'] == False)])
false_positives = len(df[(df['diameter (cm)'] <= 7) & (df[' cancer'] == True)])
true_negatives = len(df[(df['diameter (cm)'] <= 7) & (df[' cancer'] == False)])

confusion_matrix = pd.DataFrame(
    {'Actual Positive': [true_positives, false_negatives], 'Actual Negative': [false_positives, true_negatives]},
    index=['Predicted Positive', 'Predicted Negative'])

print(confusion_matrix)
print("TPR = ", tpr(true_positives, false_negatives))
print("FPR = ", fpr(false_positives, true_negatives))
print("accuracy = ", accuracy(false_positives, false_negatives, true_negatives, false_positives))
print("precision = ", precision(true_positives, false_negatives))
print("Recall = ", Recall(true_positives, false_negatives))
print("fscore = ", fscore(precision(true_positives, false_negatives), Recall(true_positives, false_negatives)))

## calculate ROC graph

sorted_probs = sorted(zip(y_true, y_pred), reverse=True)
# Initialize TPR and FPR to 0
tpr, fpr = 0, 0
# Initialize TPR and FPR lists
tpr_list, fpr_list = [0], [0]
# Iterate through sorted probabilities

for prob, true_label in sorted_probs:
    if true_label == 1:
        tpr += 1
    else:
        fpr += 1
        # Calculate TPR and FPR
        tpr_list.append(tpr / sum(y_true))
        fpr_list.append(fpr / (len(y_true) - sum(y_true)))
# Plot ROC curve
plt.plot(fpr_list, tpr_list)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
