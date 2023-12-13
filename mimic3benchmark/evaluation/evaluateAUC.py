import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import norm

# from mimic3models import metrics
from sklearn import metrics

# 根据文件名称列表csv提取样本文件（根据val_listfile.csv提取验证集）
def getFileAccList(listfile_path, oriRootPath, saveRootPath):
    if not os.path.exists(saveRootPath):
        os.makedirs(saveRootPath)
    with open(listfile_path, "r") as lfile:
        count = 0
        fileNames = lfile.readlines()[1:]
        for fileName in fileNames:
            if count % 100 == 0:
                print(count)
            fileName = fileName.replace("\n", "").split(',')[0]
            filePath = os.path.join(oriRootPath, fileName)
            savePath = os.path.join(saveRootPath, fileName)
            shutil.copyfile(filePath, savePath)
            count = count + 1

# 读死亡率预测logits汇总表，将logits与label放入list
def getLogitsAndLabelList(excelPath, modelIndex, sheetNum):
    logitsList = []
    labelList = []

    raw_data = pd.read_excel(excelPath, sheet_name=sheetNum)
    # 转array为list
    example_info_list = raw_data.values.tolist()
    for i in range(len(example_info_list)):
        logitsList.append(example_info_list[i][modelIndex])

        label_str = example_info_list[i][1]
        label_num = int(label_str)
        # label_num = [0] * len(dict_index)
        # for j in range(len(label_str)):
        #     temp_index = dict_index.index(label_str[j])
        #     label_num[temp_index] = 1
        labelList.append(label_num)
    return logitsList, labelList

def getOptimal(labelList, logitsList):
    valFprList = []
    valTprList = []
    valAucList = []
    valThrList = []
    y = np.array(labelList)
    score = np.array(logitsList)
    fpr, tpr, thr = metrics.roc_curve(y, score, pos_label=1)
    # 设置小数位数为三位
    # fpr = np.around(fpr, 3)
    # tpr = np.around(tpr, 3)
    auc = metrics.auc(fpr, tpr)

    valFprList.append(fpr.tolist())
    valTprList.append(tpr.tolist())
    valThrList.append(thr.tolist())
    valAucList.append(auc)

    # 约登指数选取最优阈值
    optimal_list = []
    for i in range(len(valTprList)):
        optimal = []
        YoudenIndexArray = np.array(valTprList[i]) + (1 - np.array(valFprList[i])) - 1
        YoudenIndexList = YoudenIndexArray.tolist()
        SortedYoudenIndexList = sorted(YoudenIndexList)
        MaxYoudenIndex = SortedYoudenIndexList[len(SortedYoudenIndexList) - 1]
        MaxYoudenIndexLoca = YoudenIndexList.index(MaxYoudenIndex)

        optimal.append(valAucList[i])
        optimal.append(valThrList[i][MaxYoudenIndexLoca])
        optimal.append(valFprList[i][MaxYoudenIndexLoca])
        optimal.append(valTprList[i][MaxYoudenIndexLoca])
        optimal.append(MaxYoudenIndex)

        optimal_list.append(optimal)

    return optimal_list, fpr, tpr, thr, auc

# 画roc_auc图像
def plotROC(fprList, tprList, aucList):
    plt.figure(figsize=(10, 10))
    plt.plot(fprList[0], tprList[0], color='red', lw=2, label='ROC curve (model of 1h, area = %0.4f)' % aucList[0])
    plt.plot(fprList[1], tprList[1], color='darkorange', lw=2, label='ROC curve (model of 2h, area = %0.4f)' % aucList[1])
    plt.plot(fprList[2], tprList[2], color='blue', lw=2, label='ROC curve (model of 4h, area = %0.4f)' % aucList[2])
    plt.plot(fprList[3], tprList[3], color='green', lw=2, label='ROC curve (model of 6h, area = %0.4f)' % aucList[3])

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Valid)')
    plt.legend(loc="lower right"),
    plt.savefig("D:\codes\mimic3-benchmarks-master/roc_valid.png")
    plt.show()
    plt.close()

def AUC_CI(auc, label, alpha = 0.05):
    label = np.array(label)#防止label不是array类型
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc / (2-auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2))
    confidence_level = 1 - alpha
    z_lower, z_upper = norm.interval(confidence_level)
    lowerb, upperb = auc + z_lower * se, auc + z_upper * se

    return (lowerb, upperb)

# 四个模型测试
if __name__ == '__main__':
    path = r'predict_result.xlsx'
    modelIndex = 2  # 改此参数，代表是第几个模型，从0开始
    sheetIndex = 0  # 改此参数，代表模型结果所在的sheet，从0开始
    # 加权[0.7, 0.9, 0.8, 0.8]
    weightIndexList = [8, 10, 9, 9] # 选择的权重预测结果所在的第几列
    valLogitsList1, valLabelList1 = getLogitsAndLabelList(path, 4, sheetIndex)
    valLogitsList2, valLabelList2 = getLogitsAndLabelList(path, 5, sheetIndex)
    valLogitsList3, valLabelList3 = getLogitsAndLabelList(path, weightIndexList[modelIndex], sheetIndex)

    optimalList1, fpr1, tpr1, thr1, auc1 = getOptimal(valLabelList1, valLogitsList1)
    optimalList2, fpr2, tpr2, thr2, auc2 = getOptimal(valLabelList2, valLogitsList2)
    optimalList3, fpr3, tpr3, thr3, auc3 = getOptimal(valLabelList3, valLogitsList3)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='blue', lw=2, label='RF-6h-MIMIC (AUC = 0.6760)')
    plt.plot(fpr2, tpr2, color='green', lw=2, label='LSTM-6h-MIMIC (AUC = %0.4f)' % auc2)
    # plt.plot(fpr3, tpr3, color='red', lw=2, label='NIMRF-6h-All (AUC = %0.4f)' % auc3)
    plt.plot(fpr3, tpr3, color='red', lw=2, label='NIMRF-6h-MIMIC (AUC = 0.6760)')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right"),
    plt.savefig(r"preResult\roc_valid_6h_%0.3f.png")
    plt.close()

    print(optimalList1)
    print(optimalList2)
    print(optimalList3)
    f = open(r'D:\articalPic\fprAndTpr_6h_0begin.txt', 'w')
    f.write("Model	Model Cut-off	TPR	TNR\n")
    for i in range(fpr1.size):
        f.write("RF\t{}\t{}\t{}\n".format(thr1[i], tpr1[i], 1 - fpr1[i]))
    for i in range(fpr2.size):
        f.write("LSTM\t{}\t{}\t{}\n".format(thr2[i], tpr2[i], 1 - fpr2[i]))
    for i in range(fpr3.size):
        f.write("RF + LSTM\t{}\t{}\t{}\n".format(thr3[i], tpr3[i], 1 - fpr3[i]))
    f.close()

# 计算AUC置信区间
# if __name__ == '__main__':
#     path = r'D:\articalPic\predict_mimic+hosp.xlsx'
#     # path = r'D:\datas\43variables\301test\result_allVariable_20221117\predict_RF+LSTM_12h.xlsx'
#     # path = r'D:\datas\43variables_models\forest\20221010\12h_mix_0.25\predict_laboratory+equipment.xlsx'
#     modelIndex = 3  # 改此参数，代表是第几个模型
#     sheetIndex = 3
#     # 论文加权[0.7, 0.9, 0.8, 0.8]
#     weightIndexList = [8, 10, 9, 9] # 选择的权重预测结果所在的第几列
#
#     # LSTM的
#     pred_prob, label = getLogitsAndLabelList(path, 5, sheetIndex)
#     FPR, TPR, _ = roc_curve(label, pred_prob, pos_label=1)
#     auc = roc_auc_score(label, pred_prob)
#     (lowerb, upperb) = AUC_CI(auc, label, alpha=0.05)
#     print("{}\t{}".format(auc, (lowerb, upperb)))
#
#     # 随机森林的
#     pred_prob, label = getLogitsAndLabelList(path, 4, sheetIndex)
#     FPR, TPR, _ = roc_curve(label, pred_prob, pos_label=1)
#     auc = roc_auc_score(label, pred_prob)
#     (lowerb, upperb) = AUC_CI(auc, label, alpha=0.05)
#     print("{}\t{}".format(auc, (lowerb, upperb)))
#
#     # 集成模型的
#     pred_prob, label = getLogitsAndLabelList(path, weightIndexList[modelIndex], sheetIndex)
#     FPR, TPR, _ = roc_curve(label, pred_prob, pos_label=1)
#     auc = roc_auc_score(label, pred_prob)
#     (lowerb, upperb) = AUC_CI(auc, label, alpha=0.05)
#     print("{}\t{}".format(auc, (lowerb, upperb)))
#
#     print("111")