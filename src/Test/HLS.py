import sys
import time

# 系统路径进入到上层目录，可以引用上层目录的库
sys.path.append("../zcm")
sys.path.append("../resource")
from Environment import *
from Graph import *
from utils import *


class Main:
    def __init__(self):
        self.env = None
        self.path = None
        self.start = -1
        self.end = -1
        self.cycle = -1
        self.length = -1
        self.deadline = -1
        self.delay = -1
        self.start_time = 0
        self.time_bound = 0

    def action(self, dfs=0):
        self.start, self.end, self.cycle, edge_mat, self.length, self.deadline = \
            self.env.translate_data_to_heuristic_inputs()
        self.path = None

        # dfs找到所有可用路由，bfs找到最短路由
        self.find_path_dfs()
        # self.find_path_bfs(edge_mat)

        assert self.path is not None
        if len(self.path) == 0:
            return False
        time_record = []
        for [edge, time_slot] in self.path:
            time_record.append(time_slot)
            edge.occupy_time_slot(time_slot, self.cycle, 1)
        delay = 0
        for i in range(1, len(time_record)):
            cur = time_record[i] - time_record[i - 1]
            if cur < 0:
                cur += self.cycle
            delay += cur
        self.delay = delay
        return True

    def find_path_dfs(self):
        self.path = []
        for edge in self.env.graph.edges.values():
            if edge.is_source_edge:
                self.dfs([edge.id])
        max_score = -1000000
        path = []
        for i in range(len(self.path)):
            if time.time() - self.start_time > self.time_bound:
                break
            cur_score = 0
            cur_delay = 0
            cur_time = 0
            cur_path = []
            for edge in self.path[i]:
                # find_time_slot_LD_old找到LD时隙，find_time_slot_fast找到最早时隙
                # 时延应该从发出时间开始算，而不是从0时刻开始算。第一跳的时延就是时隙，所以只需要从当前时间减去第一跳时隙即可
                start_time1 = -1
                offset = -1
                if len(cur_path) > 0:
                    start_time1 = cur_delay - cur_path[0][1]
                    offset = cur_path[-1][1]
                time_slot, score = edge.find_time_slot_fast(start_time1, offset, self.cycle, self.length, self.deadline - cur_delay)
                # print(start_time1, offset, time_slot)
                cur_path.append([edge, time_slot])
                if time_slot < 0:
                    cur_delay = 1000000
                    break
                cur_score += score
                cur_delay += (time_slot - cur_time + args.global_cycle) % args.global_cycle + 1
                cur_time = time_slot
            if cur_delay < self.deadline and cur_score > max_score:
                max_score = cur_score
                path = cur_path
        self.path = path

    def dfs(self, cur_path):
        # if len(cur_path) > args.network_width:
        #     return
        # print(time.time() - self.start_time)
        if time.time() - self.start_time > self.time_bound:
            return
        cur_edge_id = cur_path[-1]
        cur_edge = self.env.graph.edges[cur_edge_id]
        if cur_edge.is_destination_edge:
            temp_path = []
            for edge_id in cur_path:
                temp_path.append(self.env.graph.edges[edge_id])
            self.path.append(temp_path)
            return
        for edge in self.env.graph.edges.values():
            if edge.id not in cur_path and edge.start_node.id == cur_edge.end_node.id:
                cur_path.append(edge.id)
                self.dfs(cur_path)
                del cur_path[-1]

    def find_path_bfs(self, edge_mat):
        tree = {}
        cur_layer = []
        visited = set()
        edge_num = len(self.env.graph.edges)
        for edge in self.env.graph.edges.values():
            if edge.is_source_edge:
                cur_layer.append(edge.id)
                visited.add(edge.id)
        des = -1
        while des == -1:
            assert len(cur_layer) > 0
            next_layer = []
            for i in cur_layer:
                if self.env.graph.edges[i].is_destination_edge:
                    des = i
                    break
                for j in range(edge_num):
                    if edge_mat[i][j] == 1 and j not in visited:
                        tree[j] = i
                        next_layer.append(j)
                        visited.add(j)
            cur_layer = next_layer
        edge_path = [self.env.graph.edges[des]]
        while des in tree:
            edge_path.insert(0, self.env.graph.edges[tree[des]])
            des = tree[des]
        cur_time = 0
        self.path = []
        for edge in edge_path:
            time_slot = edge.find_time_slot_LD_old(cur_time, self.cycle, self.length, self.deadline)[0]
            if time_slot < 0:
                self.path = []
                break
            cur_time = cur_time + (time_slot - cur_time % self.cycle + self.cycle) % self.cycle + 1
            self.path.insert(0, [edge, time_slot])
        # print(self.path[0].start_node.id, self.path[-1].end_node.id, len(self.path))

    def find_time_slot_fastest(self, edge, cycle):
        time_slot = -1
        for pos in range(cycle):
            flag = True
            while pos < 1024 and flag:
                if edge.time_slot_available[pos] == 0:
                    flag = False
                pos += cycle
            if flag:
                time_slot = pos % cycle
                break
        return time_slot

    def find_time_slot_smart(self, edge):
        return edge.find_time_slot_heuristic(-1, self.cycle, self.length, self.deadline)[0]


def test(actor_agent):
    actor_agent.env = Environment()  # DataGenerater(node_num))
    actor_agent.start_time = time.time()
    cur_time = time.time()
    flow_number = 1
    info_record = {}
    while actor_agent.action():
        # print("flow_number", flow_number, "cycle", actor_agent.cycle, "time", time.time() - cur_time,
        #       "hop", len(actor_agent.path), "delay", actor_agent.delay, "usage", actor_agent.env.edge_usage())
        info_record[flow_number] = [flow_number, actor_agent.cycle, time.time() - cur_time, len(actor_agent.path), actor_agent.delay]
        actor_agent.env.enforce_next_query()
        flow_number += 1
        cur_time = time.time()
        if cur_time - actor_agent.start_time > actor_agent.time_bound:
            break

    return len(info_record), actor_agent.env.edge_usage()


def write_result_init():
    result_file_path = "Result/temp.xls"
    result_sheet = "HLS_A380"
    value_title = [["Data", "TS Number", "Link Usage", "Valid Test", "Total Time", "Time Per Flow"]]
    write_excel_xls(result_file_path, result_sheet, value_title)


def write_result(result):
    result_file_path = "Result/temp.xls"
    write_excel_xls_append(result_file_path, result)


def main():
    actor_agent = Main()
    datasets = []
    for i in range(0, 1000):
        datasets.append(f"../re/A380_NetWork/{i}/")
    actor_agent.time_bound = 600
    cur_time = time.time()
    # 创建结果文件
    write_result_init()
    for dataset in datasets:
        args.data_path = dataset
        num, usage = test(actor_agent)
        result = [args.data_path, num, usage, "", time.time() - cur_time, 0]
        # 将结果写入文件
        write_result([result])
        print(result)
        cur_time = time.time()


if __name__ == '__main__':
    main()

