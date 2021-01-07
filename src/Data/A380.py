# coding=utf-8
import json
import numpy as np
import random
import os
import sys

sys.path.append("../zcm")
from param import *

if not os.path.exists('../resource/data'):
    os.mkdir("../resource/data")

class A380Generater:
    def __init__(self):
        self.node_mat = None
        self.node_info = {}
        self.tt_flow = []
        self.tt_flow_cycle_option = args.tt_flow_cycles

    # eps 表示每条边相连的概率
    def node_mat_gene(self, dynamic=False):
        self.node_num = 8
        if dynamic:
            self.node_mat = np.zeros((self.node_num, self.node_num))
            # links = [[25, 1], [25, 22], [25, 15], [26, 12], [26, 17], [26, 20], [27, 3], [27, 5], [27, 24],
            #         [28, 9], [28, 21], [28, 18], [29, 4], [29, 10], [29, 11], [30, 6], [30, 8], [30, 14],
            #         [31, 7], [31, 23], [31, 13], [32, 2], [32, 19], [32, 16],
            #         [25, 26], [25, 27], [25, 32], [26, 28], [26, 32], [27, 29], [27, 32], [28, 30], [28, 31], [28, 32], [29, 30], [29, 31], [30, 31]]
            links = [[1, 3], [3, 5], [5, 6], [6, 4], [4, 2], [2, 1], [8, 1], [8, 2], [8, 3], [8, 4],
                     [7, 5], [7, 6], [7, 3], [7, 4], [3, 4]]
            for link in links:
                self.node_mat[link[0] - 1, link[1] - 1] = 1
                self.node_mat[link[1] - 1, link[0] - 1] = 1
        else:
            self.node_mat = np.load('../jhy/data/node_mat.npy')
        return self.node_mat

    # rand_min 最小缓存容量，rand_max 最大缓存容量
    # 生成结果{node_idx : buff_size }
    def node_info_gene(self, rand_min=30, rand_max=100, dynamic=False):
        self.rand_min = rand_min
        self.rand_max = rand_max
        if dynamic:
            self.node_info = {}
            for i in range(self.node_num):
                self.node_info[i] = random.randint(rand_min, rand_max)
        else:
            self.node_info = json.load(open('../jhy/data/node_info.json'))
        return self.node_info

    # 生成 TT 流的个数
    # delay 单位：ms、pkt_len 单位：byte
    def tt_flow_gene(self, tt_num=1, delay_min=2048, delay_max=4096, pkt_min=72, pkt_max=1526, dynamic=False):
        self.tt_num = tt_num
        if dynamic:
            self.tt_flow = []
            for i in range(tt_num):
                s = random.randint(0, self.node_num - 1)
                e = random.randint(0, self.node_num - 1)
                while e == s:
                    e = random.randint(0, self.node_num - 1)
                cycle = self.tt_flow_cycle_option[random.randint(0, len(self.tt_flow_cycle_option) - 1)]
                delay = random.randint(delay_min, delay_max)
                pkt_len = random.randint(pkt_min, pkt_max)
                self.tt_flow.append([s, e, cycle, delay, pkt_len])
        else:
            self.tt_flow = json.load(open('../jhy/data/tt_flow.json'))
        return self.tt_flow

    # 生成新调度
    def gene_all(self, rand_min = 30, rand_max = 100,
                 tt_num = 1, delay_min = 2048, delay_max = 4096, pkt_min = 72, pkt_max = 1526, dynamic = False):
        print("generate network...")
        self.node_mat_gene(dynamic=dynamic)
        self.node_info_gene(rand_min=rand_min, rand_max=rand_max, dynamic=dynamic)
        self.tt_flow_gene(tt_num=tt_num, delay_min=delay_min, delay_max=delay_max,
                          pkt_min=pkt_min, pkt_max=pkt_max, dynamic=dynamic)

        print("function A380 gene_all finish")
        return self.node_mat, self.node_info, self.tt_flow

    # 指定保存文件路径
    def write_to_file(self, filename = ""):
        if not os.path.exists(f'../jhy/{filename}'):
            os.mkdir(f'../jhy/{filename}')
        if self.node_mat is not None:
            np.save(f'../jhy/{filename}/node_mat.npy', self.node_mat)
        if self.node_info:
            json.dump(self.node_info, open(f'../jhy/{filename}/node_info.json', "w"), indent=4)
        if self.tt_flow:
            json.dump(self.tt_flow, open(f'../jhy/{filename}/tt_flow.json', "w"), indent=4)

    # 指定读取文件路径
    def read_from_file(self, filename):
        self.node_mat = np.load(f'data/{filename}/node_mat.npy')
        self.node_info = json.load(open(f'data/{filename}/node_info.json'))
        self.tt_flow = json.load(open(f'data/{filename}/tt_flow.json'))

if __name__ == '__main__':

    # data_gene.gene_all(rand_min=5, rand_max=10, tt_num=60000,
    #                    delay_min=2048, delay_max=4096, pkt_min=72, pkt_max=1526, dynamic=True)
    for i in range(1000):
        data_gene = A380Generater()
        data_gene.gene_all(rand_min=1000, rand_max=1000, tt_num=60000,
                           delay_min=64, delay_max=256, pkt_min=64, pkt_max=1526, dynamic=True)
        data_gene.write_to_file(filename=f'A380_NetWork/{i}')

    print(data_gene.node_mat)
    print(data_gene.node_info)
    # print(data_gene.tt_flow)