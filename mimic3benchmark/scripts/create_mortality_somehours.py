from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
from datetime import datetime
import random
# random.seed(49297)
random.seed()
from tqdm import tqdm

def getFilePaths(foldPath):
    filePaths = []
    for root, sub_folders, files in os.walk(foldPath):
        for file in files:
            filePath = root + '\\' + file
            filePaths.append(filePath)
    return filePaths

def dataSupplement(ts_lines, expectedBeginTime, expectedEndTime):
    # 截取到的起始时间与期望起始时间相差超过1小时，按第一条数据往前补充，每一小时补充一条
    beginTime_str = ts_lines[0].split(',')[0]
    beginTime = int(float(beginTime_str))
    if beginTime > expectedBeginTime:
        ori_pos = 0
        for i in range(beginTime, expectedBeginTime, -1):
            # addTime = round(((i - 1) + i) / 2, 4)
            addTime = ((i - 1) + i) / 2
            addTime = str(addTime)
            addData = ts_lines[ori_pos].replace(beginTime_str, addTime)
            ts_lines.insert(0, addData)
            ori_pos = ori_pos + 1

    # 截取到的终止时间与期望终止时间相差超过1小时，按最后一条数据往后补充，每小时补充一条
    endTime_str = ts_lines[len(ts_lines) - 1].split(',')[0]
    endTime = int(float(endTime_str)) + 1
    if endTime < expectedEndTime:
        ori_pos = len(ts_lines) - 1
        for i in range(endTime, expectedEndTime):
            # addTime = round(((i + 1) + i) / 2, 4)
            addTime = ((i + 1) + i) / 2
            addTime = str(addTime)
            addData = ts_lines[ori_pos].replace(endTime_str, addTime)
            ts_lines.append(addData)
    return ts_lines

def process_partition_somehours(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    xTimeInterval_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                # 在ICU停留的时间少于48小时+预测时长，删除
                if los < n_hours + int(args.interest_time_interval) - eps:
                    continue

                stay = stays_df[stays_df.ICUSTAY_ID == label_df.iloc[0]['Icustay']]
                deathtime = stay['DEATHTIME'].iloc[0]
                intime = stay['INTIME'].iloc[0]

                if pd.isnull(deathtime):
                    lived_time = 1e18
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0
                    # 有的死亡时间晚于出ICU时间，属于院内死亡，但在ICU外死亡，删除此类数据
                    if lived_time > los:
                        continue
                    # 在ICU停留时间少于48小时+预测时长，或者在ICU存活时间少于48小时+预测时长，删除
                    # elif lived_time < n_hours + int(args.interest_time_interval) - eps:
                    #     continue

                # 生成一个随机数，用于取时间序列
                # 不同时长的预测模型数据生成，主要改这个参数
                # time_interval = random.uniform(eps + int(args.interest_time_interval) - 2, int(args.interest_time_interval))
                time_interval = random.uniform(eps + 2, int(args.interest_time_interval))
                # time_interval = 2.100602

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # 如果没有死亡，提取前48小时数据，生成负样本
                if pd.isnull(deathtime):
                    ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                                if -eps < t < n_hours + eps]
                # 如果死亡，以死亡时间为节点，往前提取大约48小时数据，生成正样本
                else:
                    # ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                    #             if max(0.0, lived_time - ll - time_interval) < t < lived_time - time_interval]
                    ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                                if max(0.0, lived_time - n_hours - time_interval - eps) < t < (lived_time - time_interval + eps)]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                # 存活样本数据增补
                if pd.isnull(deathtime):
                    expectedBeginTime = 0
                    expectedEndTime = 48
                    ts_lines = dataSupplement(ts_lines,expectedBeginTime, expectedEndTime)
                # 死亡样本数据增补
                else:
                    expectedBeginTime = int(max(0.0, lived_time - n_hours - time_interval))
                    expectedEndTime = int(lived_time - time_interval) + 1
                    ts_lines = dataSupplement(ts_lines, expectedBeginTime, expectedEndTime)

                output_ts_filename = "{}_".format(args.interest_time_interval) + patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                if mortality == 0:
                    cur_mortality = 0
                else:
                    cur_mortality = 1
                xy_pairs.append((output_ts_filename, cur_mortality))
                xTimeInterval_pairs.append((output_ts_filename, time_interval))

                # xy_pairs.append((output_ts_filename, mortality))


    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))
    with open(os.path.join(output_dir, "listfile_timeInterval.csv"), "w") as listfile:
        listfile.write('stay,timeInterval\n')
        for (x, y) in xTimeInterval_pairs:
            listfile.write('{},{:f}\n'.format(x, y))

# 只保留医院历史数据库中的变量
def extractVariable(csvPath, deleteCloumnList):
    df = pd.read_csv(csvPath)
    df.head(2)
    df = df.drop(deleteCloumnList, axis=1)
    newFilePath = csvPath.replace("supple", "supple_newVariable")
    splitPath = newFilePath.split('\\')
    newFoldPath = splitPath[0] + "\\"
    for i in range(1, len(splitPath) - 1):
        newFoldPath = os.path.join(newFoldPath, splitPath[i])
    if not os.path.exists(newFoldPath):
        os.makedirs(newFoldPath)
    df.to_csv(newFilePath, index=False, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('interest_time_interval', type=str, help="Time interval of prediction.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition_somehours(args, "test")
    process_partition_somehours(args, "train")


if __name__ == '__main__':
    main()
    # pathList = getFilePaths(r'D:\datas\mortality_6h_supple')
    # deleteVariableList = ['Capillary refill rate', 'Fraction inspired oxygen', 'Glascow coma scale eye opening',
    #                       'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response',
    #                       'Mean blood pressure', 'pH']
    # for i in range(len(pathList)):
    #     if i % 500 == 0:
    #         print(i)
    #     if ("episode" in pathList[i]) and ("timeseries" in pathList[i]):
    #         extractVariable(pathList[i], deleteVariableList)
    #     else:
    #         continue
    # # extractVariable('D:\\datas\\test\\3\\episode1_timeseries.csv', )
