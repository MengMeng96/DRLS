from Node import *
from Edge import *
import json
import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from Edge import Edge

sys.path.append("../resource")
sys.path.append("../zcm")
from param import *
from Data.Ladder import *

class Graph:
    def __init__(self, data=None):
        self.adjacent_node_matrix = []
        self.adjacent_edge_matrix = []
        self.reachable_edge_matrix = []
        self.distance_edge_matrix = []
        self.start_node = []
        self.end_node = []
        self.nodes = {}
        self.edges = {}
        self.node_to_edge = {}
        self.data = data
        self.depth = 8

        # self.data_generater = DataGenerater()
        # 需要先调用 load_adjacent_matrix 生成邻接矩阵，该函数包含了生成节点个数
        self.load_adjacent_matrix()
        self.load_nodes()
        self.init()
        # print(len(self.edges))

    def init(self):
        self.get_edges()
        # 在此处更新 depth 会引起报错
        # self.update_graph_depth()
        self.get_reachable_edge_matrix()
        self.get_distance_edge_matrix()
        # self.update_graph_depth()

    def load_nodes(self):
        if self.data is None:
            info = json.load(open(args.data_path + 'node_info.json', encoding='utf-8'))
        else:
            info = self.data.node_info
        # 读取的 info 是一个字典，也可以改成下述方式
        self.nodes = {}
        for id, buffer in info.items():
            self.nodes[int(id)] = Node(int(id), buffer)

    def load_adjacent_matrix(self):
        if self.data is None:
            self.adjacent_node_matrix = np.load(args.data_path + 'node_mat.npy', allow_pickle=True)
        else:
            self.adjacent_node_matrix = self.data.node_mat

    def get_edges(self):
        id = 0
        node_len = len(self.adjacent_node_matrix)
        start_from_node = {}
        end_with_node = {}
        self.edges = {}
        self.node_to_edge = {}
        for i in range(node_len):
            start_from_node[i] = []
            end_with_node[i] = []
        for i in range(node_len):
            for j in range(node_len):
                if self.adjacent_node_matrix[i][j] == 0 or i == j:
                    continue
                self.edges[id] = Edge(id, self.nodes[i], self.nodes[j])
                self.node_to_edge[(i, j)] = id
                start_from_node[i].append(id)
                end_with_node[j].append(id)
                id += 1
        print("edge number: ", id)
        # 初始化边邻接矩阵
        self.adjacent_edge_matrix = np.zeros([id, id])
        for i in range(id):
            for j in start_from_node[self.edges[i].end_node.id]:
                self.adjacent_edge_matrix[i][j] = 1
        # for edge in self.edges.values():
        #     print(edge.id, edge.start_node.id, edge.end_node.id)
        # print(self.adjacent_edge_matrix)

    def update_graph_depth(self):
        one_step = np.array(self.adjacent_edge_matrix)
        pre_step = one_step
        depth = 1
        while depth < self.depth:
            cur_step = np.matmul(pre_step, one_step) + np.matmul(one_step, pre_step) + one_step + pre_step
            cur_step[cur_step >= 1] = 1
            if np.sum(cur_step) == np.sum(pre_step):
                break
            pre_step = cur_step
            depth += 1

        self.depth = depth
        args.max_depth = depth
        # print("graph maximum width", depth)
        return depth

    # self.reachable_edge_matrix[layer][start][cur_edge]表示start能否最短你经过layer跳到哪cur_edge
    def get_reachable_edge_matrix(self):
        edge_num = len(self.edges)
        self.reachable_edge_matrix = [np.zeros([edge_num, edge_num]) for _ in range(self.depth)]
        for start in range(edge_num):
            # print(start)
            visited_node = {start}
            cur_list = [start]
            for layer in range(self.depth):
                next_list = []
                for cur_edge in cur_list:
                    self.reachable_edge_matrix[layer][start][cur_edge] = 1
                    for adjacent in range(edge_num):
                        if self.adjacent_edge_matrix[cur_edge][adjacent] == 1 and \
                                adjacent not in visited_node:
                            visited_node.add(adjacent)
                            next_list.append(adjacent)
                cur_list = next_list
        # for i in range(self.depth):
        #     print(i, self.reachable_edge_matrix[i])
        #     print(i, np.sum(self.reachable_edge_matrix[i]))

    def get_distance_edge_matrix(self):
        edge_num = len(self.edges)
        node_num = len(self.nodes)
        self.distance_edge_matrix = np.full([edge_num, node_num], 9999)
        for i in range(edge_num):
            visited_edge = {i}
            cur_list = [self.edges[i]]
            dis = 1
            self.distance_edge_matrix[i][self.edges[i].start_node.id] = 0
            while len(cur_list) > 0:
                next_list = []
                for cur_edge in cur_list:
                    self.distance_edge_matrix[i][cur_edge.end_node.id] = min(self.distance_edge_matrix[i][cur_edge.end_node.id], dis)
                    for adjacent in range(edge_num):
                        if self.adjacent_edge_matrix[cur_edge.id][adjacent] == 1 and adjacent not in visited_edge:
                            visited_edge.add(adjacent)
                            next_list.append(self.edges[adjacent])
                cur_list = next_list
                dis += 1

    def delete_edge(self, edge_id, tt_flow_to_edge):
        self.adjacent_node_matrix[edge_id[0]][edge_id[1]] = 0
        self.init()
        for tt_flow in tt_flow_to_edge.values():
            for info in tt_flow:
                edge_tuple = info[0]
                time_slot = info[1]
                cycle = info[2]
                length = info[3]
                edge_id = self.node_to_edge[edge_tuple]
                # 因为所有的数据都重置了，所以要把所有的流再导入一遍
                self.edges[edge_id].occupy_time_slot(time_slot, cycle, length)


    def reset(self):
        for edge in self.edges.values():
            edge.reset()
        for node in self.nodes.values():
            node.reset()


def main():
    graph = Graph()
    print("node")
    for i in range(len(graph.nodes)):
        print(i, end=": ")
        for j in range(len(graph.nodes)):
            if graph.adjacent_node_matrix[i][j] == 1:
                print(j, end=' ')
        print()
    print("edge")
    for edge in graph.edges.values():
        print(edge.id, edge.start_node.id, edge.end_node.id)
    graph.edges[0].occupy_time_slot(0, 256, 1)
    graph.edges[0].occupy_time_slot(1, 256, 1)
    graph.edges[0].find_time_slot(0, 0, 128, 1, 1)
    # for c in graph.edges[0].time_slot_status:
    #     print(graph.edges[0].time_slot_status[c])
    print(graph.node_to_edge)


if __name__ == '__main__':
    main()