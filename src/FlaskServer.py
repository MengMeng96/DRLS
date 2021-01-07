import string

import flask
from flask import request
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")
sys.path.append("TorchVersion")

from flask import Flask  # 引入Flask模块
from DRLS import *
from Agent import *
from pytorch_op import *
import time
import pickle

scheduler = Flask(__name__) # 创建一个应用


@scheduler.route('/', methods = ["GET","POST"])
def index():    # 定义根目录处理器resource
    return '<h1>Hello World!</h1>'

# http://127.0.0.1:8053/bunchCalculate
@scheduler.route('/bunchCalculate', methods = ["GET","POST"])
def bunchCalculateAPI():
    node_info = {"0": 9, "1": 8, "2": 8, "3": 10, "4": 7}
    node_matrix = [[1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 1],
                   [1, 0, 1, 1, 1],
                   [0, 0, 1, 1, 0],
                   [0, 1, 1, 0, 1]]

    print(node_matrix)
    tt_flow = [[0, 1, 64, 85, 1032],
                [1, 2, 64, 249, 112],
                [1, 4, 64, 184, 88],
                [3, 2, 128, 343, 769],
                [3, 2, 256, 366, 962],
                [3, 2, 512, 247, 579],
                [1, 2, 256, 496, 1289],
                [3, 4, 512, 140, 679],
                [0, 1, 256, 293, 1380],
                [2, 1, 128, 140, 871]]

    # System init
    np.random.seed()

    # Agent init
    environment, information = bunchCalculate(node_info, node_matrix, tt_flow)

    f = open('Result/variables/environment.pickle', 'wb')
    pickle.dump(environment, f)
    f.close()
    return information

# http://127.0.0.1:8053/getBunchCalculateResult
@scheduler.route('/getBunchCalculateResult', methods = ["GET","POST"])
def getBunchCalculateResultAPI():
    f = open('Result/variables/environment.pickle', 'rb')
    environment = pickle.load(f)
    f.close()
    return getBunchCalculateResult(environment)

#http://127.0.0.1:8053/singleCalculate?src=0&dst=2&cycle=512&ddl=140&length=871
@scheduler.route('/singleCalculate', methods = ["GET","POST"])
def singleCalculateAPI():
    query = [0, 2, 512, 140, 871]

    f = open('Result/variables/environment.pickle', 'rb')
    environment = pickle.load(f)
    f.close()

    environment, information = singleCalculate(environment, query)

    f = open('Result/variables/environment.pickle', 'wb')
    pickle.dump(environment, f)
    f.close()
    return information


@scheduler.route('/getSingleCalculateResult', methods = ["GET","POST"])
def getSingleCalculateResultAPI():
    f = open('Result/variables/environment.pickle', 'rb')
    environment = pickle.load(f)
    f.close()
    return getSingleCalculateResult(environment)


schedule = None
if __name__ == '__main__':
    schedule = None
    scheduler.run(debug=True,host='0.0.0.0', port=8053) # 启动服务
