from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib
import joblib
import pandas as pd
import random
from numpy import *
from train_mortality import extractDataAndLabel

from IPython.display import Image
from sklearn import tree
import pydotplus
import os
import pylab
import matplotlib.pyplot as plt
from matplotlib import image
import cv2

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
        fLog = open("outputs\\test\\predict_" + str(index) + ".txt", "w")

        for i in range(len(testData)):
            predict = model.predict(testData[i].reshape((1, -1)))
            predictLogits = model.predict_proba(testData[i].reshape((1, -1)))
            fLog.write(str(predict))
            fLog.write('\n')
        fLog.close()
    fLogForest = open("outputs\\test\\predict_Forest.txt", "w")
    for i in range(len(testData)):
        predict = rf.predict(testData[i].reshape((1, -1)))
        predictLogits = rf.predict_proba(testData[i].reshape((1, -1)))
        fLogForest.write(testCube[i][0])
        fLogForest.write('\t')
        fLogForest.write(str(predict))
        fLogForest.write('\t')
        fLogForest.write(str(predictLogits))
        fLogForest.write('\n')
    fLogForest.close()

# 测试模型准确率
def countAccuracy():
    count = 0
    # 打印识别结果
    fLog = open("outputs\\test\\predicts.txt", "w")
    for i in range(len(testData)):
    #     testData[i] = array(testData[i])
    #     testData[i] = testData[i].flatten()
        if testLabel[i][0] == model.predict(testData[i].reshape((1, -1))):
            count = count + 1
        fLog.write(testCube[i][0])
        fLog.write('\t')
        fLog.write(str(testLabel[i][0]))
        fLog.write('\t')
        fLog.write(str(model.predict(testData[i].reshape((1, -1))))[1:-1])
        fLog.write('\t')
        fLog.write(str(model.predict_proba(testData[i].reshape((1, -1))))[1:-1])
        fLog.write('\n')
    accuracy = float(count / len(testData))
    accuracy = round(accuracy, 5)
    fLog.write(str(accuracy))
    fLog.write('\n')
    print("accuracy:" + str(accuracy))

# 决策树可视化
def visualRandomTree(rf):
    print("visual decisionTrees...")
    # 提取一个决策树
    Estimators = rf.estimators_
    for index, model in enumerate(Estimators):
        filename = 'outputs\\test\\visual\\decisionTree_' + str(index) + '.pdf'
        dot_data = tree.export_graphviz(model, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # 使用ipython的终端jupyter notebook显示。
        Image(graph.create_png())
        graph.write_pdf(filename)

# 随机森林中各决策树特征权重可视化
def featureImportanceDecisionTree(rf):
    print("evaluate feature importance...")
    Estimators = rf.estimators_
    for index, model in enumerate(Estimators):
        fLog = open("outputs\\test\\visual\\featureImportance_" + str(index) + ".txt", "w")
        for i in range(len(testData)):
            if testLabel[i][0] == model.predict(testData[i].reshape((1, -1))):
                count = count + 1
            fLog.write(testCube[i][0])
            fLog.write('\t')
            fLog.write(str(testLabel[i][0]))
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
        importance = importance.reshape(31, 13)
        # plt.matshow(importance)
        plt.matshow(importance, cmap=plt.cm.hot)
        plt.title("Pixel importances with tree" + str(index))
        plt.savefig("outputs\\test\\visual\\featureImportance_" + str(index) + ".png")

# 随机森林各特征权重可视化
def featureImportance(rf):
    print("evaluate feature importance...")
    fLog = open("outputs\\test\\visual\\featureImportance.txt", "w")
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
    plt.savefig("outputs\\test\\visual\\featureImportance.png")
    plt.show()

# 数据样本可视化
def visualData(data_1331_index):
    plt.matshow(testData[data_1331_index])
    plt.title(testCube[i][0] + "_" + str(testLabel[i][0]))
    saveDir = "outputs\\trainData\\" + str(testLabel[i][0]) + "\\"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + testCube[i][0] + "_" + str(testLabel[i][0]) + ".png")
    # plt.show()

if __name__ == '__main__':
    # 测试
    # print("predict...")
    testDataTemp, testLabel, testCube = extractDataAndLabel("inputs\\PicInfo.xlsx")
    testData = []
    for m in range(len(testDataTemp)):
        testDataFill = []
        if len(testDataTemp[m]) == 25:
            for i in range(3):
                testDataFill.append(testDataTemp[m][0])
            for j in range(len(testDataTemp[m])):
                testDataFill.append(testDataTemp[m][j])
            for k in range(3):
                testDataFill.append(testDataTemp[m][24])
        else:
            testDataFill = testDataTemp[m]
        testData.append(testDataFill)
    for i in range(len(testData)):
        visualData(i)
        testData[i] = array(testData[i])
        testData[i] = testData[i].flatten()

    # 加载模型
    with open("model.pkl", "rb") as f:
        modelRF = joblib.load(f)

    # 打印各决策树与随机森林的预测结果
    # predictTreeAndForest(model)

    # 计算准确率
    countAccuracy()
    # 决策树可视化
    visualRandomTree(model)
    # 评估特征重要性
    featureImportance(model)
    featureImportanceDecisionTree(model)
    featureImportance(rf)