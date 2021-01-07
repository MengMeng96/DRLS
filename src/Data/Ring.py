# coding=utf-8
import json
import numpy as np
import random
import os
import sys
sys.path.append("../zcm")
# if not os.path.exists('../jhy/data'):
#     os.mkdir("../jhy/data")
from param import *


class RingGenerater:
    def __init__(self):
        self.link_info = None
        self.node_num = 0
        self.node_info = {}
        self.tt_flow = []
        self.tt_flow_cycle_option = args.tt_flow_cycles

    # eps 表示每条边相连的概率
    def link_info_gene(self, eps = 0.5, dynamic = False):
        # id : {topologyId:int,
        # sourceNodeId:int, sourcePort:int, destNodeId:int, destPort:int,
        # status:int, cfgBandwidth:int, latency:int}
        self.link_info = {}
        for i in range(self.node_num):
            if i % 2 == 0:
                a = int(i / 2)
                b = (a + 1) % self.node_num
            else:
                b = int(i / 2)
                a = (a + 1) % self.node_num
            self.link_info[i] = {"id": i,
                                 "topologyId": 0,
                                 "sourceNodeId": a,
                                 "sourcePort": i % 2,
                                 "destNodeId": b,
                                 "destPort": (i + 1) % 2,
                                 "status": 1,
                                 "cfgBandwidth": 1000,
                                 "latency": 0}

    # rand_min 最小缓存容量，rand_max 最大缓存容量
    # 生成结果{node_idx : buff_size }
    def node_info_gene(self, rand_min=30, rand_max=100, dynamic = False):
        self.node_info = {}
        print(self.node_num)
        for i in range(self.node_num):
            self.node_info[i] = {"id": i,
                                 "topologyId": 0,
                                 "name": "switch",
                                 "status": 1,
                                 "type": 0,
                                 "gclCount": 256,
                                 "tsBusinessQueueCount": 4}

    # 生成 TT 流的个数
    # delay 单位：ms、pkt_len 单位：byte
    def tt_flow_gene(self, tt_num = 1, delay_min = 2048, delay_max = 4096, pkt_min = 72, pkt_max = 1526, dynamic = False):
        self.tt_flow = {}
        for i in range(tt_num):
            s = random.randint(0, self.node_num - 1)
            e = random.randint(0, self.node_num - 1)
            while e == s:
                e = random.randint(0, self.node_num - 1)
            cycle = self.tt_flow_cycle_option[random.randint(0, len(self.tt_flow_cycle_option) - 1)]
            delay = random.randint(delay_min, delay_max)
            pkt_len = random.randint(pkt_min, pkt_max)
            self.tt_flow[i] = {"id": i,
                               "name": "flow",
                               "maxFrameSize": pkt_len,
                               "period": cycle,
                               "talkerID": s,
                               "listenerIDs": [e],
                               "maxLatency": delay,
                               "jitter": 0}
        return self.tt_flow

    # 生成新调度
    def gene_all(self, node_num = 10, eps = 0.2,
                 rand_min = 30, rand_max = 100,
                 tt_num = 1, delay_min = 2048, delay_max = 4096, pkt_min = 72, pkt_max = 1526,
                 hop = 1, dynamic = False):
        self.node_num = node_num
        self.link_info_gene(eps=eps, dynamic=dynamic)
        self.node_info_gene(rand_min=rand_min, rand_max=rand_max, dynamic=dynamic)
        self.tt_flow_gene(tt_num=tt_num, delay_min=delay_min, delay_max=delay_max,
                          pkt_min=pkt_min, pkt_max=pkt_max, dynamic=dynamic)

        print("function Ring gene_all finish")
        return self.link_info, self.node_info, self.tt_flow

    # 指定保存文件路径
    def write_to_file(self, filename=""):
        if not os.path.exists(f'../resource/{filename}'):
            os.mkdir(f'../resource/{filename}')
        json.dump(self.link_info, open(f'../resource/{filename}/link_info.json', "w"), indent=4)
        json.dump(self.node_info, open(f'../resource/{filename}/node_info.json', "w"), indent=4)
        json.dump(self.tt_flow, open(f'../resource/{filename}/tt_flow.json', "w"), indent=4)


if __name__ == '__main__':
    data_gene = RingGenerater()
    if not os.path.exists(f'../resource/PCL_NetWork/test'):
        os.mkdir(f'../resource/PCL_NetWork/test')
    for i in range(0, 1):
        node_num = 10
        data_gene.gene_all(node_num=node_num, eps=0.35, rand_min=5, rand_max=10, tt_num=8,
                           delay_min=64, delay_max=512, pkt_min=72, pkt_max=1526, hop=1, dynamic=True)

        data_gene.write_to_file(filename=f"PCL_NetWork/test/{i}")

    # print(data_gene.node_mat)
    # print(data_gene.node_info)
    # print(data_gene.tt_flow)