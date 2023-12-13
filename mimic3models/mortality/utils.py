from __future__ import absolute_import
from __future__ import print_function

from mimic3models import common_utils
import numpy as np
import os


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    print("read_chunk...")
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    # for i in range(len(data)):
    #     data[i] = data[i].flatten()

    # print("write data...")
    # for i in range(len(data)):
    #     # .xlsx改成.txt
    #     txtname = names[i].replace('.xlsx', '.txt')
    #     # txtname = names[i].replace('.csv', '.txt')
    #
    #     f = open(r'D:\datas\43variables\forest\301\alltest\{}'.format(txtname), 'w')
    #     data_list = data[i].tolist()
    #     for j in range(len(data_list)):
    #         data_row = str(data_list[j]).replace('[','').replace(']', '').replace(', ', '\t')
    #         f.write(data_row)
    #         f.write('\n')
    #     # f.write(str(data_list))
    #     f.close()
    #     if i%500 == 0:
    #         print(i)
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
