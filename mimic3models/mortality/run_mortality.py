from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib
import joblib
import pandas as pd
import random
from numpy import *
from train import extractDataAndLabel

from IPython.display import Image
from sklearn import tree
import pydotplus
import os
import pylab
import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import numpy as np
from sklearn import preprocessing

def getFilePaths(foldPath):
    filePaths = []
    for root, sub_folders, files in os.walk(foldPath):
        for file in files:
            filePath = root + '/' + file
            filePaths.append(filePath)
    return filePaths

def extractDataInfo(dataInfoPath):
    dataInfo = pd.read_excel(dataInfoPath, sheet_name=0)

    allData = []
    allLabel = []
    allName = []
    # 转array为list
    pic_info_list = dataInfo.values.tolist()
    for i in range(len(pic_info_list)):
        if i % 100 == 0:
            print(i)
        # dataTxtPath = 'D:\\datas\\43variables\\forest\\20221010\\mortality_20221010\\train\\{}.txt'.format(pic_info_list[i][0][:-4])
        dataTxtPath = 'D:\\datas\\43variables\\forest\\301\\all_20221117\\{}.txt'.format(pic_info_list[i][0][:-5])
        dataLabel = pic_info_list[i][1]
        dataName = pic_info_list[i][0]
        with open(dataTxtPath, 'r') as f:
            data = f.readlines()
            for j in range(len(data)):
                data[j] = eval('[{}]'.format(data[j].replace('\t', ',')))
            data = np.array(data)

            # 数据归一化
            max_abs_scaler = preprocessing.MaxAbsScaler()
            dataStan = max_abs_scaler.fit_transform(data)
            allData.append(dataStan)
            # imp = SimpleImputer(missing_values=np.nan, strategy='median', verbose=0, copy=True)
            # dataFill = imp.fit_transform(data)
            # allData.append(dataFill)
        allLabel.append(dataLabel)
        allName.append(dataName)
    return allData, allLabel, allName

# 检查是否为整套数据
def checkDataCompleted(allPic_info_path):
    raw_data = pd.read_excel(allPic_info_path)
    # 转array为list
    pic_info_list = raw_data.values.tolist()
    f = open("outputs\\非整套数据.txt", "w")
    for i in range(len(pic_info_list)):
        if pic_info_list[i][4] == "000.jpg":
            picNum = int(pic_info_list[i][2])
            rightTail = picNum - 1
            if pic_info_list[i + picNum - 1][4] != "0" + str(rightTail) + ".jpg":
                print(pic_info_list[i][3])
                f.write(pic_info_list[i][3])
                f.write('\n')
        if i%100 == 0:
            print(i)

# 打印每棵树及整个森林预测结果
def predictTreeAndForest(rf):
    Estimators = rf.estimators_
    for index, model in enumerate(Estimators):
        print("predict according Tree" + str(index))
        fLog = open("D:\\datas\\forest\\preResult\\predict_" + str(index) + ".txt", "w")

        for i in range(len(testData)):
            predict = model.predict(testData[i].reshape((1, -1)))
            predictLogits = model.predict_proba(testData[i].reshape((1, -1)))
            fLog.write(testName[i])
            fLog.write('\t')
            fLog.write(str(predict))
            fLog.write('\t')
            fLog.write(str(predictLogits))
            fLog.write('\n')
        fLog.close()
    fLogForest = open("D:\\datas\\forest\\preResult\\predict_Forest.txt", "w")
    for i in range(len(testData)):
        predict = rf.predict(testData[i].reshape((1, -1)))
        predictLogits = rf.predict_proba(testData[i].reshape((1, -1)))
        fLogForest.write(testName[i])
        fLogForest.write('\t')
        fLogForest.write(str(predict))
        fLogForest.write('\t')
        fLogForest.write(str(predictLogits))
        fLogForest.write('\n')
    fLogForest.close()

# 测试模型性能
def countAccuracyE(modelRF, modelName):
    # 打印识别结果
    # fLog = open(r"D:\datas\43variables_models\forest\20221010\12h_mix_0.25\preResult_valid_all\predict_{}.txt".format(modelName), "w")
    fLog = open(r"D:\datas\43variables\301test\result_allVariable_20221117\RF_test_predictions_allVariable_6h_24.29.txt".format(modelName), "w")
    # fLog = open(r"D:\datas\43variables_models\forest\20221010\6h_mix_0.25\preResult_valid_all\predict_model_6h_20221012_estimators_24_random_29.txt".format(modelName), "w")
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(testData)):
        # if i % 200 == 0:
        #     print(i)
        modelPredict = modelRF.predict(testData[i].reshape((1, -1)))
        modelPredictProba = modelRF.predict_proba(testData[i].reshape((1, -1)))

        if modelPredict[0] == 1 and testLabel[i] == 1:
            TP = TP + 1
        elif modelPredict[0] == 1 and testLabel[i] == 0:
            FP = FP + 1
        elif modelPredict[0] == 0 and testLabel[i] == 0:
            TN = TN + 1
        else:
            FN = FN + 1

        fLog.write(testName[i])
        fLog.write('\t')
        fLog.write(str(testLabel[i]))
        fLog.write('\t')
        fLog.write(str(modelPredict)[1:-1])
        fLog.write('\t')
        fLog.write(str(modelPredictProba[0].tolist()[0]))
        fLog.write('\t')
        fLog.write(str(modelPredictProba[0].tolist()[1]))
        # fLog.write('\t')
        # fLog.write(str(modelPredictProba[0].tolist()[2]))
        # fLog.write('\t')
        # fLog.write(str(modelPredictProba[0].tolist()[3]))
        # fLog.write('\t')
        # fLog.write(str(modelPredictProba[0].tolist()[4]))
        fLog.write('\n')
        # fLog.write(str(modelPredictProba)[2:-2].split(' ')[0])
        # fLog.write('\t')
        # fLog.write(str(modelPredictProba)[2:-2].split(' ')[1])
        # fLog.write('\t')
        # fLog.write(str(modelPredictProba)[2:-2].split(' ')[2])
        # fLog.write('\t')
        # fLog.write(str(modelPredictProba)[2:-2].split(' ')[3])
        # fLog.write('\t')
        # fLog.write(str(modelPredictProba)[2:-2].split(' ')[4])
        # fLog.write('\n')
    fLog.close()

    TPR = float(TP / (TP + FN))
    TNR = float(TN / (TN + FP))
    # TNR = 0
    ACC = float((TP + TN) / len(testData))
    fAccE = open(r"D:\datas\43variables\301test\result_allVariable_20221117\predict_accuracyE.txt", "a+")
    # fAccE = open(r"D:\datas\43variables_models\forest\20221010\12h_mix_0.25\preResult_valid_all\predict_accuracyE.txt","a+")
    fAccE.write("{}\t{}\t{}\t{}\n".format(modelName, round(TPR, 5), round(TNR, 5), round(ACC, 5)))
    fAccE.close()
    print("{}\t{}\t{}\t{}\n".format(modelName, round(TPR, 5), round(TNR, 5), round(ACC, 5)))

# 决策树可视化
def visualRandomTree(rf):
    print("visual decisionTrees...")
    # 提取一个决策树
    Estimators = rf.estimators_
    for index, model in enumerate(Estimators):
        # filename = 'outputs\\test\\20190108\\visual_model_14.7_delete\\decisionTree_' + str(index) + '.pdf'
        filename = r'D:\software\part2\OCT_entityClassification\outputs\test\20220908\visual_model\decisionTree_' + str(index) + '.pdf'
        dot_data = tree.export_graphviz(model, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # 使用ipython的终端jupyter notebook显示。
        Image(graph.create_png())
        graph.write_pdf(filename)

# 随机森林中各决策树特征权重可视化
def featureImportanceDecisionTree(rf):
    print("evaluate feature importance...")
    Estimators = rf.estimators_
    count = 0
    for index, model in enumerate(Estimators):
        # fLog = open("outputs\\test\\20190108\\visual_feature_14.7_delete\\featureImportance_" + str(index) + ".txt", "w")
        fLog = open(r"D:\software\part2\OCT_entityClassification\outputs\test\20220908\visual_feature\featureImportance_" + str(index) + ".txt", "w")
        for i in range(len(testData)):
            #     testData[i] = array(testData[i])
            #     testData[i] = testData[i].flatten()
            if testLabel[i] == model.predict(testData[i].reshape((1, -1))):
                count = count + 1
            fLog.write(testName[i])
            fLog.write('\t')
            fLog.write(str(testLabel[i]))
            fLog.write('\t')
            fLog.write(str(model.predict(testData[i].reshape((1, -1))))[1:-1])
            fLog.write('\t')
            fLog.write(str(model.predict_proba(testData[i].reshape((1, -1))))[1:-1])
            fLog.write('\n')
        importance = model.feature_importances_
        # indices = argsort(importance)[::-1]
        for i in range(len(testData[0])):
            fLog.write(str(int(i / 13)))
            fLog.write('\t')
            fLog.write(str(i))
            fLog.write('\t')
            # fLog.write(str(importance[indices[i]]))
            fLog.write(str(importance[i]))
            fLog.write('\n')
        importance = importance.reshape(50, 97)
        # plt.matshow(importance)
        plt.matshow(importance, cmap=plt.cm.hot)
        plt.title("Pixel importances with tree" + str(index))
        # plt.savefig("outputs\\test\\20190108\\visual_feature_14.7_delete\\featureImportance_" + str(index) + ".png")
        plt.savefig(r"D:\software\part2\OCT_entityClassification\outputs\test\20220908\visual_feature\featureImportance_" + str(index) + ".png")
        # plt.show()

        # plt.subplot(479)
        # title = 'importance'
        # plt.title(title)
        # plt.imshow(importance)
        # pylab.show()

# 数据可视化
def visualData(data_1331_index):
    plt.matshow(testData[data_1331_index])
    plt.title(str(testLabel[i]))
    saveDir = 'D:\\datas\\forest\\visual_stan_301\\{}\\'.format(str(testLabel[i]))
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    # print(111)
    plt.savefig(saveDir + testName[i]+ ".png")
    plt.close()

# 随机森林各特征权重可视化
def featureImportance(rf):
    print("evaluate feature importance...")
    # fLog = open("outputs\\test\\20190108\\visual_feature_14.7_delete\\featureImportance.txt", "w")
    fLog = open(r"D:\articalPic\ModelExpla\RF\featureImportance_1h.txt", "w")
    importance = rf.feature_importances_
    # indices = argsort(importance)[::-1]
    for i in range(len(testData[0])):
        fLog.write(str(int(i/13)))
        fLog.write('\t')
        fLog.write(str(i))
        fLog.write('\t')
        # fLog.write(str(importance[indices[i]]))
        fLog.write(str(importance[i]))
        fLog.write('\n')
        print(importance[i])
    importance = importance.reshape(31, 13)
    # Plot pixel importances
    plt.matshow(importance, cmap=plt.cm.hot)
    plt.title("Pixel importances with forests of trees")
    plt.savefig("outputs\\test\\20190108\\visual_feature_14.7_delete\\featureImportance.png")
    plt.show()
    # plt.subplot(133)
    # title = 'importance'
    # plt.title(title)
    # # cv2.imwrite("outputs\\test\\20190103\\featureImportance.png", importance)
    # image.imsave("outputs\\test\\20190103\\featureImportance.png", importance)
    # plt.imshow(importance)
    # pylab.show()

# 数据样本可视化
# def visualData(data_1331_index):
#     plt.matshow(testData[data_1331_index])
#     plt.title(testName[i] + "_" + str(testLabel[i][0]))
#     saveDir = "outputs\\trainData\\" + str(testLabel[i][0]) + "\\"
#     if not os.path.exists(saveDir):
#         os.makedirs(saveDir)
#     plt.savefig(saveDir + testName[i] + "_" + str(testLabel[i][0]) + ".png")
#     # plt.show()

def main1():
    # 测试
    # print("predict...")
    # testData, testLabel, testName = extractDataInfo(r"D:\datas\43variables\mimic\mortality_2h\val_listfile.xlsx")
    # testData, testLabel, testName = extractDataInfo(r"D:\datas\43variables\mimic\mortality_2h\test_listfile.xlsx")
    for i in range(len(testData)):
        # visualData(i)
        # trainData[i] = array(trainData[i])
        testData[i] = testData[i].flatten()

    # modelPaths = getFilePaths(r'D:\datas\43variables\301test\allDatarun')
    modelPaths = getFilePaths(r'D:\datas\43variables_models\forest\20221010\12h_mix_0.25\run')
    for i in range(len(modelPaths)):
        with open(modelPaths[i], "rb") as f:
            modelRF = joblib.load(f)
        countAccuracyE(modelRF, modelPaths[i].split('/')[-1][:-4])
    # 加载模型
    # with open("D:\\datas\\forest_6h\\model\\model_20220918_estimators_33_random_8.pkl", "rb") as f:
    #      modelRF = joblib.load(f)

    # 打印各决策树与随机森林的预测结果
    # predictTreeAndForest(modelRF)

    # 计算准确率
    # countAccuracyE(modelRF, 'model_20220917_estimators_26_random_27')
    # 决策树可视化
    # visualRandomTree(modelRF)
    # 评估特征重要性
    # featureImportance(modelRF)
    # featureImportanceDecisionTree(modelRF)

def main2():
    # 测试
    # print("predict...")
    # testData, testLabel, testName = extractDataInfo(r"D:\datas\43variables\mimic\mortality_2h\val_listfile.xlsx")
    # testData, testLabel, testName = extractDataInfo(r"D:\datas\43variables\mimic\mortality_2h\test_listfile.xlsx")
    print("flatten data...")
    for i in range(len(testData)):
        if i % 100 == 0:
            print(i)
        testData[i] = testData[i].flatten()

    # 加载模型
    print("load model...")
    with open(r"D:\datas\43variables_models\forest\20221010\00_models\model_1h_20221012_estimators_26_random_12.pkl", "rb") as f:
    # with open(r"D:\datas\43variables_models\forest\20221010\00_models\model_6h_20221012_estimators_24_random_29.pkl", "rb") as f:
    # with open(r"D:\datas\43variables_models\forest\6h\run\model_6h_20220920_estimators_39_random_2.pkl", "rb") as f:
    # with open(r"D:\datas\43variables_models\forest\4h\run\model_4h_20220920_estimators_35_random_2.pkl", "rb") as f:
    # with open(r"D:\datas\43variables_models\forest\1h\run\model_1h_20220920_estimators_33_random_29.pkl", "rb") as f:
    # with open(r"D:\datas\43variables_models\forest\2h\run\model_20220920_estimators_27_random_7.pkl", "rb") as f:
         modelRF = joblib.load(f)

    # 打印各决策树与随机森林的预测结果
    # predictTreeAndForest(modelRF)

    # 计算准确率
    print("count acc...")
    countAccuracyE(modelRF, 'model_6h_20221012_estimators_24_random_29')
    # 决策树可视化
    visualRandomTree(modelRF)
    # 评估特征重要性
    featureImportance(modelRF)
    # featureImportanceDecisionTree(modelRF)
if __name__ == '__main__':
    print("extract data...")
    # testData, testLabel, testName = extractDataInfo(r"D:\datas\43variables\mimic\20221010\mortality_20221010\val_listfile_all_1h.xlsx")
    testData, testLabel, testName = extractDataInfo(r"D:\datas\43variables\301test\allData\test_listfile_haveData_1h.xlsx")
    # 验证集找最优模型
    # main1()
    # 最优模型跑测试集
    main2()