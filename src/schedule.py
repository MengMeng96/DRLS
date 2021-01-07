# coding=utf-8
import json
import numpy as np
import os

import sys
# sys.path.append("../jhy")

SCHE_FILE = '../resource/data-2-hop/schedule.json'

if not os.path.exists('../resource/data'):
    os.mkdir("../resource/data")

class Schedule:
    def __init__(self, node_num = 0):
        self.node_num = node_num
        self.sche = {}

    # 对于一条流进行调度时的初始化，这些信息不会发生变化
    # 起点IP、TT流编号、周期、报文长度、保留位1、保留位2
    def sche_start(self, start_IP, tt_idx, cycle = 1024, length = 128, info_a = 0, info_b = 0):
        self.start_IP = start_IP
        self.tt_idx = tt_idx
        self.cycle = cycle
        self.length = length
        self.info_a = info_a
        self.info_b = info_b

    def find_port(self, start_IP, des_IP):
        # 这块需要端口和DAG的信息，先默认取 1 进行测试
        return 1, 1
        pass

    def reset(self):
        self.sche = {}

    def delete_by_tt_flow_id(self, tt_idx):
        for node in self.sche.values():
            for port in node.values():
                for directions in port.values():
                    if tt_idx in directions:
                        directions.pop(tt_idx)

    def update(self, IP, offset):
        if IP == -1 or offset == -1:
            # self.write_json()
            # self.reset()
            return self.sche
        # 从 self.start_IP 节点的 send_port 端口发送报文至 IP 节点的 recv_port 端口
        send_port, recv_port = IP, self.start_IP  # self.find_port(self.start_IP, IP)
        tt2info = {}
        tt2info["cycle"] = self.cycle
        tt2info["offset"] = offset
        tt2info["length"] = self.length
        tt2info["info_a"] = self.info_a
        tt2info["info_b"] = self.info_b
        if self.start_IP not in self.sche:
            self.sche[self.start_IP] = {}
        IP2port = self.sche[self.start_IP]
        if send_port not in IP2port:
            IP2port[send_port] = {}
        if "send" not in IP2port[send_port]:
            IP2port[send_port]["send"] = {}
        port2flow = IP2port[send_port]["send"]
        port2flow[self.tt_idx] = tt2info

        tt2info = {}
        tt2info["cycle"] = self.cycle
        tt2info["offset"] = offset
        tt2info["length"] = self.length
        tt2info["info_a"] = self.info_a
        tt2info["info_b"] = self.info_b
        if IP not in self.sche:
            self.sche[IP] = {}
        IP2port = self.sche[IP]
        if recv_port not in IP2port:
            IP2port[recv_port] = {}
        if "receive" not in IP2port[recv_port]:
            IP2port[recv_port]["receive"] = {}
        port2flow = IP2port[recv_port]["receive"]
        port2flow[self.tt_idx] = tt2info

        self.start_IP = IP
        return self.sche

    # 注：json文件中字典的键值都为字符串类型
    def write_json(self, filename = SCHE_FILE):
        json.dump(self.sche, open(filename, "w"), indent=4)

    def read_json(self, filename = SCHE_FILE):
        self.sche = json.load(open(filename))

    # npy 不适合字典类型的存取，使用 json 文件保存调度文件
    # def write_npy(self):
    #     np.save('schedule.npy', self.sche)

    # npy 不适合字典类型的存取，使用 json 文件保存调度文件
    # def read_npy(self):
    #     self.sche = np.load('schedule.npy')

    def show(self):
        for id in self.sche:
            print("id: ", id)
            for port in self.sche[id]:
                print("    port: ", port)
                for dir in self.sche[id][port]:
                    print("        direction: ", dir)
                    for k in self.sche[id][port][dir]:
                        print("            ", k, ": ", self.sche[id][port][dir][k])

    def toString(self):
        str = ""
        for id in self.sche:
            str = str + "\n" + f"id: {id}"
            for port in self.sche[id]:
                str = str + f"\n    port: {port}"
                for dir in self.sche[id][port]:
                    str = str + f"\n        direction: {dir}"
                    for k in self.sche[id][port][dir]:
                        str = str + f"\n            stream {k}" + ": " + f"{self.sche[id][port][dir][k]['offset']}"
        return str


if __name__ == '__main__':
    ss = Schedule(10)
    # 开始调度一条流
    # start_IP, tt_idx, cycle, length, info_a, info_b
    ss.sche_start('1.1.1.1', 1, 4, 128, 0, 0)
    # 下一个往 '1.1.1.2' 节点发送，slot 为3
    ss.update('1.1.1.2', 3)
    ss.update('1.1.1.3', 5)
    ss.update('1.1.1.5', 13)

    ss.write_json()
    ss.read_json()

    # 开始调度另一条流
    ss.sche_start('1.1.1.5', 3, 8, 256, 0, 0)
    ss.update('1.1.1.4', 13)
    ss.update('1.1.1.2', 24)
    ss.update('1.1.1.3', 37)
    ss.update('1.1.1.1', 56)

    ss.write_json()
