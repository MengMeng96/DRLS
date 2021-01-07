# coding=utf-8
import json
import numpy as np
import random
import os
import sys
sys.path.append("../zcm")
if not os.path.exists('../resource/data'):
    os.mkdir("../resource/data")
from param import *


class DataGenerater:
    def __init__(self):
        self.node_mat = None
        self.node_num = 0
        self.node_info = {}
        self.tt_flow = []
        self.tt_flow_cycle_option = args.tt_flow_cycles
        # self.gene_all(node_num=node_num, eps=0.3, rand_min=1000, rand_max=1000, tt_num=60000, hop=1, dynamic=True)

    # eps 表示每条边相连的概率
    def node_mat_gene(self, node_num = 10, eps = 0.5, dynamic = False):
        assert node_num > 1, "node_num is too small"
        self.node_num = node_num
        self.eps = eps
        if dynamic:
            self.node_mat = np.random.random((node_num, node_num))
            for i in range(node_num):
                # 对角线默认为 1
                self.node_mat[i, i] = 1
                for j in range(i):
                    self.node_mat[i, j] = 1 if self.node_mat[i, j] < eps else 0
                    self.node_mat[j, i] = self.node_mat[i, j]
        else:
            self.node_mat = np.load('../resource/data/node_mat.npy')
        return self.node_mat

    # rand_min 最小缓存容量，rand_max 最大缓存容量
    # 生成结果{node_idx : buff_size }
    def node_info_gene(self, rand_min = 30, rand_max = 100, dynamic = False):
        self.rand_min = rand_min
        self.rand_max = rand_max
        if dynamic:
            self.node_info = {}
            for i in range(self.node_num):
                self.node_info[i] = random.randint(rand_min, rand_max)
        else:
            self.node_info = json.load(open('../resource/data/node_info.json'))
        return self.node_info

    # 生成 TT 流的个数
    # delay 单位：ms、pkt_len 单位：byte
    def tt_flow_gene(self, tt_num = 1, delay_min = 2048, delay_max = 4096, pkt_min = 72, pkt_max = 1526, dynamic = False):
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
            self.tt_flow = json.load(open('../resource/data/tt_flow.json'))
        return self.tt_flow

    # 生成新调度
    def gene_all(self, node_num = 10, eps = 0.2,
                 rand_min = 30, rand_max = 100,
                 tt_num = 1, delay_min = 2048, delay_max = 4096, pkt_min = 72, pkt_max = 1526,
                 hop = 1, dynamic = False):
        reachable = False
        print("generate network...")
        self.node_num = node_num
        while not reachable:
            # print(".", end='')
            self.node_mat_gene(node_num = node_num, eps = eps, dynamic = dynamic)
            self.node_info_gene(rand_min = rand_min, rand_max = rand_max, dynamic = dynamic)
            self.tt_flow_gene(tt_num = tt_num, delay_min = delay_min, delay_max = delay_max,
                              pkt_min = pkt_min, pkt_max = pkt_max, dynamic = dynamic)

            reachable = True
            for i in range(self.node_num):
                for j in range(i):
                    reachable = reachable and self.is_reachable(i, j, hop = hop)
                    if not reachable:
                        break
        print("function gene_all finish")
        return self.node_mat, self.node_info, self.tt_flow

    def is_reachable(self, start, end, hop = 2):
        nodes = [start]
        s = set()
        cnt = 1
        while nodes:
            tmp = []
            for node in nodes:
                for i in range(self.node_num):
                    if self.node_mat[node][i] and i not in s:
                        s.add(i)
                        tmp.append(i)
            if end in tmp:
                if cnt >= hop:
                    return True
                else:
                    return False
            nodes = tmp
            cnt += 1
        return False

    # 指定保存文件路径
    def write_to_file(self, filename = ""):
        if not os.path.exists(f'../resource/{filename}'):
            os.mkdir(f'../resource/{filename}')
        if self.node_mat is not None:
            np.save(f'../resource/{filename}/node_mat.npy', self.node_mat)
        if self.node_info:
            json.dump(self.node_info, open(f'../resource/{filename}/node_info.json', "w"), indent=4)
        if self.tt_flow:
            json.dump(self.tt_flow, open(f'../resource/{filename}/tt_flow.json', "w"), indent=4)

    # 指定读取文件路径
    def read_from_file(self, filename):
        self.node_mat = np.load(f'data/{filename}/node_mat.npy')
        self.node_info = json.load(open(f'data/{filename}/node_info.json'))
        self.tt_flow = json.load(open(f'data/{filename}/tt_flow.json'))


if __name__ == '__main__':
    data_gene = DataGenerater()
    for n in range(5, 6):
        if not os.path.exists(f'../resource/Random_NetWork/PCL'):
            os.mkdir(f'../resource/Random_NetWork/PCL')
        for i in range(0, 10):
            data_gene.gene_all(node_num=n, eps=0.35, rand_min=5, rand_max=10, tt_num=10 * i + 10,
                               delay_min=64, delay_max=512, pkt_min=72, pkt_max=1526, hop=1, dynamic=True)

            data_gene.write_to_file(filename=f"Random_NetWork/PCL/{i}")

    # print(data_gene.node_mat)
    # print(data_gene.node_info)
    # print(data_gene.tt_flow)