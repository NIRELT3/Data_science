import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def tpr(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def recall(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def fpr(false_positive, true_negative):
    return false_positive / (false_positive + true_negative)


def precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)


def fscore(precision, recall):
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1


def accuracy(true_positive, false_negative, true_negative, false_positive):
    return (true_positive + true_negative) / (true_positive + false_negative + true_negative + false_positive)


df = pd.read_csv(r'C:\Users\galmo\training\DS_cancer\CANCER_TABLE.csv')


def get_data() -> pd.DataFrame:
    cancerdf = pd.read_csv(os.path.join('data_dir', 'CANCER_TABLE.csv'))
    return cancerdf


def main():
    df = get_data()
    df_filtered = df[df['diameter (cm)'] > 7]

    true_positives = len(df_filtered[(df_filtered[' cancer'] == True)])
    false_negatives = len(df_filtered[(df_filtered[' cancer'] == False)])
    false_positives = len(df[(df['diameter (cm)'] <= 7) & (df[' cancer'] == True)])
    true_negatives = len(df[(df['diameter (cm)'] <= 7) & (df[' cancer'] == False)])

    confusion_matrix = pd.DataFrame(
        {'Actual Positive': [true_positives, false_negatives], 'Actual Negative': [false_positives, true_negatives]},
        index=['Predicted Positive', 'Predicted Negative'])
    print(confusion_matrix)
    # print("TPR = ", tpr(true_positives, false_negatives))
    # print("FPR = ", fpr(false_positives, true_negatives))
    print("accuracy = ", accuracy(false_positives, false_negatives, true_negatives, false_positives))
    print("precision = ", precision(true_positives, false_negatives))
    print("Recall = ", recall(true_positives, false_negatives))

    ## calculate ROC graph without sklearn
    thresholds = list(np.array(list(range(0, 100+1, 5)))/10)
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        tp = len(df[(df['diameter (cm)'] > threshold) & (df[' cancer'] == True)])
        fn = len(df[(df['diameter (cm)'] < threshold) & (df[' cancer'] == True)])
        tn = len(df[(df['diameter (cm)'] < threshold) & (df[' cancer'] == False)])
        fp = len(df[(df['diameter (cm)'] > threshold) & (df[' cancer'] == False)])

        tpr1 = tp / (tp + fn)
        fpr1 = fp / (fp + tn)
        tpr_list.append(tpr1)
        fpr_list.append(fpr1)
    print(tpr_list)
    print("----")
    print(fpr_list)

    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * tpr_list[i]
    print('AUC:', auc)
    # plot ROC curve
    plt.scatter(fpr_list, tpr_list)
    # plt.plot(fpr_list, tpr_list, label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(df[' cancer'], df['diameter (cm)'])

    # calculate AUC
    # auc_value = auc(fpr, tpr)
    # print('AUC:', auc_value)

    # plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' )
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
