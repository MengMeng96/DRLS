import sys
import time

# 系统路径进入到上层目录，可以引用上层目录的库
sys.path.append("../zcm")
sys.path.append("../jhy")
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
        self.reschedule_cnt = -1
        self.reschedule_start_time = -1
        self.reschedule_end_time = -1

    def action(self, dfs=0):
        self.start, self.end, self.cycle, edge_mat, self.length, self.deadline = \
            self.env.translate_data_to_heuristic_inputs()
        self.path = None

        # dfs找到所有可用路由，bfs找到最短路由
        # self.find_path_dfs()
        self.find_path_bfs(edge_mat)

        assert self.path is not None
        # print(self.path)
        if len(self.path) == 0:
            return False
        time_record = []
        for [edge, time_slot] in self.path:
            time_record.append(time_slot)
            # edge.occupy_time_slot(time_slot, self.cycle, 1)
            self.env.step(edge, time_slot, 0, True)
        delay = 0
        for i in range(1, len(time_record)):
            cur = time_record[i] - time_record[i - 1]
            if cur < 0:
                cur += self.cycle
            delay += cur
        self.delay = delay
        return True

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
            # 时延应该从发出时间开始算，而不是从0时刻开始算。第一跳的时延就是时隙，所以只需要从当前时间减去第一跳时隙即可
            start_time = -1
            offset = -1
            if len(self.path) > 0:
                start_time = cur_time - self.path[0][1]
                offset = self.path[-1][1]
            time_slot, score = edge.find_time_slot(start_time, offset, self.cycle, self.length, self.deadline)
            # print(time_slot, score, self.deadline)
            # time_slot = edge.find_time_slot_LD_old(cur_time, self.cycle, self.length, self.deadline)[0]
            if time_slot < 0:
                self.path = []
                break
            cur_time = cur_time + (time_slot - cur_time % args.global_cycle + args.global_cycle) % args.global_cycle + 1
            self.path.append([edge, time_slot])
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
    start_time = time.time()
    flow_number = 1
    info_record = {}
    while actor_agent.action():
        end_time = time.time()
        #print("flow_number", flow_number, "cycle", actor_agent.cycle, "time", end_time - start_time,
        #      "hop", len(actor_agent.path), "delay", actor_agent.delay, "usage", actor_agent.env.edge_usage())
        info_record[flow_number] = [flow_number, actor_agent.cycle, end_time - start_time, len(actor_agent.path), actor_agent.delay]
        if flow_number == args.link_failure_pos:
            actor_agent.reschedule_start_time = time.time()
            edge = actor_agent.env.graph.edges[0]
            actor_agent.reschedule_cnt = len(actor_agent.env.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)])
            # print(actor_agent.env.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)], actor_agent.reschedule_start_time)
            actor_agent.env.delete_edge((edge.start_node.id, edge.end_node.id))
        if actor_agent.env.reschedule == 2:
            actor_agent.reschedule_end_time = time.time()
        actor_agent.env.enforce_next_query()
        flow_number += 1
        start_time = time.time()

    return len(info_record), actor_agent.env.edge_usage(), \
           actor_agent.reschedule_end_time - actor_agent.reschedule_start_time, actor_agent.reschedule_cnt


def write_result_init():
    result_file_path = "Result/temp.xls"
    result_sheet = "LS+LD_A380"
    value_title = [["Data", "TS Number", "Link Usage", "Valid Test", "Total Time", "Time Per Flow",
                    "reschedule time", "reschedule count"]]
    write_excel_xls(result_file_path, result_sheet, value_title)


def write_result(result):
    result_file_path = "Result/temp.xls"
    write_excel_xls_append(result_file_path, result)


def main():
    actor_agent = Main()
    datasets = []
    for i in range(1, 100):
        datasets.append(f"../jhy/A380_NetWork/{i}/")
    cur_time = time.time()
    # 创建结果文件
    write_result_init()
    for dataset in datasets:
        args.data_path = dataset
        num, usage, reschedule_time, reschedule_cnt = test(actor_agent)
        result = [args.data_path, num, usage, "", time.time() - cur_time, 0, reschedule_time, reschedule_cnt]
        # 将结果写入文件
        write_result([result])
        print(result)
        cur_time = time.time()


if __name__ == '__main__':
    main()

