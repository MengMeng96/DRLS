#!/usr/bin/python
# -*- coding: UTF-8 -*-

from Node import *
from Graph import *
import json
import numpy as np
import random
import sys
import copy
import math

# sys.path.append("../resource/")
from schedule import Schedule
from param import *


class Environment:
    def __init__(self, data=None):
        self.graph = Graph(data)
        self.time = 0
        self.depth = 8
        self.decay = 0.5
        self.tt_query_id = -1
        self.reschedule_queries = []
        self.reschedule_pos = -1
        self.cur_tt_flow_id = -1
        self.reschedule = 0

        # 记录已经访问过的节点
        self.valid_edges = {}
        self.visited_node = []
        # 记录当前TT流的周期
        self.tt_flow_cycle = args.global_cycle
        self.tt_flow_start = -1
        self.tt_flow_end = -1
        self.delay = -1
        self.tt_flow_lenth = 1
        self.tt_flow_deadline = -1
        self.tt_flow_length = -1
        # 记录当前TT流的所有发出时间
        self.tt_flow_time_record = []
        # 记录每条流使用的链路和时隙
        self.tt_flow_to_edge = {}
        # 记录每个链路包含的流
        self.edge_to_tt_flow = {}
        for i in range(len(self.graph.nodes)):
            for j in range(len(self.graph.nodes)):
                self.edge_to_tt_flow[(i, j)] = set()

        # 奖励值参数
        self.stop_parameter = 10
        self.delay_parameter = 0.000001
        self.lasting_parameter = 0

        self.data = data
        if self.data is None:
            self.tt_queries = json.load(open(args.data_path + 'tt_flow.json', encoding='utf-8'))
        else:
            # 此处有 6 万条 TT-调度信息
            self.tt_queries = self.data.tt_flow

        self.schedule = Schedule()
        self.current_stream_schedule = Schedule()
        self.enforce_next_query()

    def edge_usage(self):
        total = len(self.graph.edges) * args.global_cycle
        cur = 0
        for edge in self.graph.edges.values():
            cur += sum(edge.time_slot_available)
        return cur / total

    def enforce_next_query(self, rool_back=False):
        for node in self.graph.nodes.values():
            node.is_source_node = 0
            node.is_destination_node = 0

        # print(self.reschedule_pos, self.reschedule_queries)
        if self.reschedule_pos + 1 < len(self.reschedule_queries):
            self.reschedule_pos += 1
            self.cur_tt_flow_id = self.reschedule_queries[self.reschedule_pos]
            if self.reschedule == 0:
                self.reschedule = 2
        elif not rool_back:
            self.tt_query_id += 1
            self.cur_tt_flow_id = self.tt_query_id
            self.reschedule = 0
        else:
            self.cur_tt_flow_id = self.tt_query_id

        # print(self.tt_queries[self.cur_tt_flow_id])

        self.tt_flow_start = self.tt_queries[self.cur_tt_flow_id][0]
        self.tt_flow_end = self.tt_queries[self.cur_tt_flow_id][1]
        self.tt_flow_cycle = self.tt_queries[self.cur_tt_flow_id][2]
        self.tt_flow_deadline = self.tt_queries[self.cur_tt_flow_id][3]
        self.tt_flow_length = self.tt_queries[self.cur_tt_flow_id][4]

        # 将秒转换成时隙
        self.tt_flow_deadline = int(self.tt_flow_deadline * args.slot_per_millisecond)
        self.tt_flow_cycle = int(self.tt_flow_cycle * args.slot_per_millisecond)
        # 当前报文横跨的 slot 数量，用2**17代替10**6/8
        self.tt_flow_length = int(
            math.ceil(self.tt_flow_lenth * 1.0 / (args.link_rate * 2 ** 17) * args.slot_per_millisecond))

        # print("query", self.cur_tt_flow_id, self.tt_flow_start, self.tt_flow_end, self.tt_flow_cycle,
        #       self.tt_flow_deadline, self.tt_flow_lenth)

        self.graph.nodes[self.tt_flow_start].set_source_node()
        self.graph.nodes[self.tt_flow_end].set_destination_node()
        self.visited_node = [self.tt_flow_start]
        self.tt_flow_time_record = [-1]
        self.tt_flow_to_edge[self.cur_tt_flow_id] = []

        for edge in self.graph.edges.values():
            edge.refresh()

        # 开始进行调度
        self.schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)
        self.current_stream_schedule.reset()
        self.current_stream_schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)

    def enforce_specific_query(self, query, rool_back=False):
        for node in self.graph.nodes.values():
            node.is_source_node = 0
            node.is_destination_node = 0

        self.tt_query_id += 1
        self.cur_tt_flow_id = self.tt_query_id
        if len(self.tt_queries) <= self.cur_tt_flow_id:
            self.tt_queries.append(query)
            self.tt_queries[self.cur_tt_flow_id] = query
        else:
            self.tt_queries[self.cur_tt_flow_id] = query

        # print(self.tt_queries[self.cur_tt_flow_id])

        self.tt_flow_start = self.tt_queries[self.cur_tt_flow_id][0]
        self.tt_flow_end = self.tt_queries[self.cur_tt_flow_id][1]
        self.tt_flow_cycle = self.tt_queries[self.cur_tt_flow_id][2]
        self.tt_flow_deadline = self.tt_queries[self.cur_tt_flow_id][3]
        self.tt_flow_length = self.tt_queries[self.cur_tt_flow_id][4]

        # 将秒转换成时隙
        self.tt_flow_deadline = int(self.tt_flow_deadline * args.slot_per_millisecond)
        self.tt_flow_cycle = int(self.tt_flow_cycle * args.slot_per_millisecond)
        # 当前报文横跨的 slot 数量，用2**17代替10**6/8
        self.tt_flow_length = int(
            math.ceil(self.tt_flow_lenth * 1.0 / (args.link_rate * 2 ** 17) * args.slot_per_millisecond))

        # print("query", self.cur_tt_flow_id, self.tt_flow_start, self.tt_flow_end, self.tt_flow_cycle,
        #       self.tt_flow_deadline, self.tt_flow_lenth)

        self.graph.nodes[self.tt_flow_start].set_source_node()
        self.graph.nodes[self.tt_flow_end].set_destination_node()
        self.visited_node = [self.tt_flow_start]
        self.tt_flow_time_record = [-1]
        self.tt_flow_to_edge[self.cur_tt_flow_id] = []

        for edge in self.graph.edges.values():
            edge.refresh()

        # 开始进行调度
        self.schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)
        self.current_stream_schedule.reset()
        self.current_stream_schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)


    def translate_data_to_heuristic_inputs(self):
        edge_mat = self.graph.adjacent_edge_matrix
        return self.tt_flow_start, self.tt_flow_end, self.tt_flow_cycle, edge_mat, \
               self.tt_flow_length, self.tt_flow_deadline

    def translate_data_to_inputs(self):
        self.valid_edges = {}
        edge_num = len(self.graph.edges)
        policy_inputs = np.zeros([edge_num, args.policy_input_dim])
        time_inputs = 0#np.zeros([edge_num, args.time_input_dim])
        for edge in self.graph.edges.values():
            if edge.start_node.is_source_node and edge.end_node.id not in self.visited_node \
                    and len(edge.time_slot_status[self.tt_flow_cycle]) > 0:
                edge.is_source_edge = 1
                self.valid_edges[edge.id] = edge
            if edge.end_node.is_destination_node:
                edge.is_destination_edge = 1
        total_usage = self.edge_usage()
        for edge in self.graph.edges.values():
            # is_destination = edge.is_destination_edge
            is_source = edge.is_source_edge
            # 时延应该 从发出时间算，而不是从0时刻开始算
            start_time = -1
            offset = -1
            if len(self.tt_flow_time_record) > 1:
                start_time = self.tt_flow_time_record[-1] - self.tt_flow_time_record[1]
                offset = self.tt_flow_time_record[-1] % args.global_cycle
            time_slot, score = edge.find_time_slot(start_time, offset, self.tt_flow_cycle, self.tt_flow_length,
                                                   self.tt_flow_deadline)
            if time_slot < 0:
                is_source = 0 # 将超时因素也考虑进去
            time_slot_num = self.avaiable_time_slot_number(edge)
            # policy_inputs[edge.id, 0] = is_destination
            # policy_inputs[edge.id, 1] = end_node_buffer
            # policy_inputs[edge.id, 2] = time_slot_num
            policy_inputs[edge.id, 0] = self.graph.distance_edge_matrix[edge.id, self.tt_flow_end]
            policy_inputs[edge.id, 1] = is_source  # 不是邻边也可能是因为没有可用时隙
            policy_inputs[edge.id, 2] = edge.end_node.id in self.visited_node
            policy_inputs[edge.id, 3] = time_slot_num / self.tt_flow_cycle
            policy_inputs[edge.id, 4] = sum(edge.time_slot_available) / args.global_cycle - total_usage
            policy_inputs[edge.id, 5] = (score + 120) / 120
            policy_inputs[edge.id, 6] = (self.tt_flow_deadline - self.time) / self.tt_flow_deadline
            policy_inputs[edge.id, 7] = sum(self.graph.reachable_edge_matrix[1][edge.id]) / len(self.graph.edges) * 3
            # print((score + 120) / 120, (self.tt_flow_delay - self.time) / self.tt_flow_delay, sum(self.graph.reachable_edge_matrix[1][edge.id]) / 5)
        return self.valid_edges, policy_inputs, time_inputs, self.tt_flow_cycle, self.time % args.global_cycle, \
               self.tt_flow_length, self.tt_flow_deadline

    def accumulated_buffer_occupation_rate(self, node):
        occupation_rate = 0
        decay_t = 1
        time_t = self.time % args.global_cycle
        # 此处为什么循环 depth 次？
        for i in range(self.depth):
            occupation_rate += decay_t * node.buffer_avaiable[time_t]
            time_t = (time_t + 1) % args.global_cycle
            decay_t *= self.decay
        return occupation_rate

    def avaiable_time_slot_number(self, edge):
        return len(edge.time_slot_status[self.tt_flow_cycle])

    # 返回奖励值和是否结束，1表示当前TT流调度结束，0表示调度未结束，-1表示调度失败
    def step(self, edge, time_slot, LD_score, heuristic=False):
        # print(LD_score)
        if time_slot > -1:
            self.schedule.update(edge.end_node.id, time_slot)
            self.current_stream_schedule.update(edge.end_node.id, time_slot)
        else:
            self.schedule.update(-1, -1)
            self.current_stream_schedule.update(-1, -1)

        reason = "Success"
        # 调度失败
        if (time_slot < 0 or edge.id not in self.valid_edges) and not heuristic:
            reward = -1 * self.stop_parameter - LD_score / 24 - \
                     1 / (self.tt_query_id + 1) * self.lasting_parameter
            if time_slot == -1:
                reason = "No buffer"
            elif time_slot == -2:
                # total_edge_usage = self.edge_usage()
                # usages = []
                # for edge in self.graph.edges.values():
                #     usages.append(sum(edge.time_slot_available) / args.global_cycle)
                # print(sum(edge.time_slot_available) / args.global_cycle, min(usages), total_edge_usage)
                reason = "No time slot"
                reward -= 10
                # print(reason, reward)
            elif time_slot == -3:
                reason = "timeout"
            elif edge.id not in self.valid_edges:
                reason = "Visited edge or Not adjacent edge"
                reward -= 10
            # print(reason, reward, self.valid_edges)
            self.time = 0
            return reward, -1, reason

        # 无论调度是否结束，都需要占用当前时隙
        edge.occupy_time_slot(time_slot, self.tt_flow_cycle, self.tt_flow_length)

        # 无论调度是否结束，都需要记录发出时间
        offset = self.time % args.global_cycle
        self.time = self.time + (time_slot - offset + args.global_cycle) % args.global_cycle + 1
        self.tt_flow_time_record.append(self.time)
        # 将当前选中的边加入集合，此后不能再次进入这个节点
        self.visited_node.append(edge.end_node.id)
        # 记录链路的流的信息
        self.tt_flow_to_edge[self.cur_tt_flow_id].append(
            [(edge.start_node.id, edge.end_node.id), time_slot, self.tt_flow_cycle, self.tt_flow_length])
        self.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)].add(self.cur_tt_flow_id)

        # 调度成功
        if edge.end_node.is_destination_node:
            delay = self.tt_flow_time_record[-1] - self.tt_flow_time_record[1]
            reward = 1 * self.stop_parameter - delay * self.delay_parameter - 0.1 * len(self.tt_flow_time_record) + \
                     10 * (sum(edge.time_slot_available) / args.global_cycle - self.edge_usage()) - LD_score / 24
            # 记录上一个调度
            # self.last_schedule = copy.deepcopy(self.schedule.sche)
            self.time = 0
            self.visited_node = []
            return reward, 1, reason

        # 中间步骤，更新节点和边信息
        for node in self.graph.nodes.values():
            node.is_source_node = 0
        edge.end_node.is_source_node = 1
        # print(edge.start_node.id, edge.end_node.id)
        for edge in self.graph.edges.values():
            edge.refresh()
        return 0, 0, reason

    def roll_back(self, number):
        self.reschedule_queries = []
        self.reschedule_pos = -1
        for i in reversed(range(number)):
            cur_id = self.cur_tt_flow_id - i
            if cur_id not in self.tt_flow_to_edge:
                print(cur_id)
                continue
            self.delete_tt_flow(cur_id)
            for info in self.edge_to_tt_flow.values():
                if cur_id in info:
                    info.remove(cur_id)
        self.tt_query_id = self.cur_tt_flow_id
        self.reschedule = 1

    def delete_tt_flow(self, tt_flow_id, reschedule=True):
        # print("delete flow", tt_flow_id)
        for info in self.tt_flow_to_edge[tt_flow_id]:
            edge_tuple = info[0]
            edge_id = self.graph.node_to_edge[edge_tuple]
            time_slot = info[1]
            cycle = info[2]
            self.graph.edges[edge_id].reset_time_slot(time_slot, cycle)

        self.schedule.delete_by_tt_flow_id(tt_flow_id)
        self.current_stream_schedule.reset()
        self.tt_flow_to_edge.pop(tt_flow_id)

        if not reschedule:
            for info in self.edge_to_tt_flow.values():
                if tt_flow_id in info:
                    info.remove(tt_flow_id)

        if reschedule:
            self.reschedule_queries.append(tt_flow_id)

    def delete_edge(self, edge_id):
        print(self.edge_to_tt_flow[edge_id])
        for tt_flow_id in self.edge_to_tt_flow[edge_id]:
            self.delete_tt_flow(tt_flow_id)
        self.edge_to_tt_flow[edge_id] = set()
        self.graph.delete_edge(edge_id, self.tt_flow_to_edge)

    def delete_node(self, node_id):
        node_num = len(self.graph.nodes)
        for i in range(node_num):
            if self.graph.adjacent_node_matrix[i][node_id] == 1:
                self.delete_edge((i, node_id))
            if self.graph.adjacent_node_matrix[node_id][i] == 1:
                self.delete_edge((node_id, i))

    def reset(self):
        self.graph.reset()
        self.schedule.reset()
        self.current_stream_schedule.reset()
        self.tt_query_id = 0
        self.enforce_next_query()
        self.visited_node = []


def main():
    env = Environment(Graph())
    env.enforce_next_query()
    valid_edges, policy_inputs, time_inputs = env.translate_data_to_inputs()
    for edge in valid_edges:
        print(edge.id, edge.is_source_edge, edge.time_slot_available)
    print(policy_inputs)
    print(time_inputs)


if __name__ == '__main__':
    main()
