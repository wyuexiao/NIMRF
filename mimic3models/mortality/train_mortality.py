from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.externals import joblib
import joblib
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
import pandas as pd
import random
from numpy import *
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import preprocessing

# 检查是否为整套数据
def checkDataCompleted(allPic_info_path):
    raw_data = pd.read_excel(allPic_info_path)
    # 转array为list
    pic_info_list = raw_data.values.tolist()
    for i in range(len(pic_info_list)):
        if pic_info_list[i][4] == "000.jpg":
            picNum = int(pic_info_list[i][2])
            rightTail = picNum - 1
            if pic_info_list[i + picNum - 1][4] != "0" + str(rightTail) + ".jpg":
                print(pic_info_list[i][3])
        if i%100 == 0:
            print(i)

def extractDataAndLabel(allPic_info_path):
    raw_data = pd.read_excel(allPic_info_path)
    # 转array为list
    pic_info_list = raw_data.values.tolist()
    dataSet = []
    labelSet = []
    cubeNameSet = []
    for i in range(len(pic_info_list)):
        # if i%1000 == 0:
        #     print("picNum:" + str(i))
        if pic_info_list[i][4] == "000.jpg":
            cubeData = []
            cubeLabel = []
            cubeName = []
            picNum = int(pic_info_list[i][2])
            for j in range(picNum):
                if pic_info_list[i][3] == pic_info_list[i+j][3]:
                    labelPic = pic_info_list[i + j][8]
                    logitsPic = eval(pic_info_list[i + j][7])
                    # for insertNum in range(3):
                    #     logitsPic.insert(0, logitsPic[6])
                    #     del logitsPic[7]
                    # for appendNum in range(3):
                    #     logitsPic.append(logitsPic[7])
                    #     del logitsPic[7]
                    cubeNamePic = pic_info_list[i + j][3]
                    cubeData.append(logitsPic)
                    if j == 0:
                        cubeLabel.append(labelPic)
                        cubeName.append(cubeNamePic)
                else:
                    print("cube error:" + pic_info_list[i + j][3])
            dataSet.append(cubeData)
            labelSet.append(cubeLabel)
            cubeNameSet.append(cubeName)
    return dataSet, labelSet, cubeNameSet

# 数据可视化
def visualData(data_1331_index):
    plt.matshow(trainData[data_1331_index])
    plt.title(str(trainLabel[i]))
    saveDir = 'visual_stan\\{}\\'.format(str(trainLabel[i]))
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    # print(111)
    plt.savefig(saveDir + trainName[i]+ ".png")
    plt.close()

def extractDataInfo(dataInfoPath):
    dataInfo = pd.read_excel(dataInfoPath)

    allData = []
    allLabel = []
    allName = []
    # 转array为list
    pic_info_list = dataInfo.values.tolist()
    random.shuffle(pic_info_list)
    for i in range(len(pic_info_list)):
        if i % 500 == 0:
            print("extractDataInfo: {}/{}".format(i, len(pic_info_list)))
        dataTxtPath = 'mortality\\{}.txt'.format(pic_info_list[i][0][:-4])
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

if __name__ == '__main__':
    trainData, trainLabel, trainName = extractDataInfo(r"mortality\train_listfile.xlsx")
    for i in range(len(trainData)):
        # visualData(i)
        # trainData[i] = array(trainData[i])
        trainData[i] = trainData[i].flatten()

    for i in range(23, 50):
        for j in range(30):
            rf = RandomForestClassifier(n_estimators=i, oob_score=True, random_state=j, max_features='log2',
                                        bootstrap=True)
            rf.fit(trainData, trainLabel)
            joblib.dump(rf, r'model_estimators_{}_random_{}.pkl'.format(i, j))
            print('model_estimators_{}_random_{}.pkl'.format(i, j))

    # # 训练
    # rf = RandomForestClassifier(n_estimators=14, oob_score=True, random_state=7, max_features='log2', bootstrap=True)
    # # rf = ExtraTreesClassifier(n_estimators=28, oob_score=False, random_state=5, max_features='auto')
    # rf.fit(trainData, trainLabel)
    # # 保存模型
    # joblib.dump(rf, 'model_20220116_14.7_delete_fill.pkl')
    # print("model_20220116_21.27_delete.pkl")
