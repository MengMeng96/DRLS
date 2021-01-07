import sys
# 系统路径进入到上层目录，可以引用上层目录的库
sys.path.append("../zcm")
sys.path.append("../resource")
from zcm.Environment import *
from zcm.Graph import *
import time

class Main:
    def __init__(self):
        self.env = None
        self.path = None
        self.cycle = -1
        self.delay = -1

    def action(self, dfs=0):
        start, end, cycle, edge_mat = self.env.translate_data_to_heuristic_inputs()
        self.path = None
        if dfs:
            for edge in self.env.graph.edges.values():
                if edge.is_source_edge:
                    # print(edge.id)
                    self.find_path_dfs([edge], [edge.id])
        else:
            self.find_path_bfs(edge_mat)
        assert self.path is not None
        self.cycle = cycle
        time_record = []
        for edge in self.path:
            time_slot = self.find_time_slot_smart(edge, cycle)
            time_record.append(time_slot)
            if time_slot < 0:
                return False
            # edge.occupy_time_slot(time_slot, cycle)
            self.env.step(edge, time_slot, heuristic=True)
        delay = 0
        for i in range(1, len(time_record)):
            cur = time_record[i] - time_record[i - 1]
            if cur < 0:
                cur += cycle
            delay += cur
        self.delay = delay
        return True

    def find_path_dfs(self, cur_path, visited_edge):
        if self.path is not None:
            return
        cur_edge = cur_path[-1]
        if cur_edge.is_destination_edge:
            if self.path is None or len(cur_path) < len(self.path):
                self.path = cur_path
        for edge in self.env.graph.edges.values():
            if edge.start_node.id == cur_edge.end_node.id and \
                    edge.id not in visited_edge:
                cur_path.append(edge)
                visited_edge.append(edge.id)
                self.find_path_dfs(cur_path, visited_edge)
                cur_path = cur_path[0: -1]
                visited_edge = visited_edge[0: -1]

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
        self.path = [self.env.graph.edges[des]]
        while des in tree:
            self.path.insert(0, self.env.graph.edges[tree[des]])
            des = tree[des]
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

    def find_time_slot_smart(self, edge, cycle):
        time_slot, _ = edge.find_time_slot(-1, cycle)
        return time_slot


def main():
    agent = Main()
    node_nums = [15 for _ in range(1)]
    for node_num in node_nums:
        agent.env = Environment() # DataGenerater(node_num))
        start_time = time.time()
        flow_number = 1
        info_record = {}
        root_time = time.time()
        reschedule_cnt = -1
        while agent.action():
            end_time = time.time()
            print("flow_number", flow_number, "cycle", agent.cycle, "time", end_time - start_time, "hop", len(agent.path), "delay", agent.delay, "total time", time.time() - root_time)
            info_record[flow_number] = [flow_number, agent.cycle, end_time - start_time, len(agent.path), agent.delay]
            if flow_number == args.link_failure_pos:
                root_time = time.time()
                edge = agent.env.graph.edges[0]
                reschedule_cnt = len(agent.env.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)])
                print(agent.env.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)])
                agent.env.delete_edge((edge.start_node.id, edge.end_node.id))
            agent.env.enforce_next_query()
            flow_number += 1
            start_time = time.time()
            # if agent.env.tt_query_id == args.link_failure_pos:
            #     break
        print(agent.env.edge_usage(), ',', len(info_record))
        print(reschedule_cnt, time.time() - root_time, (time.time() - root_time) / reschedule_cnt)


if __name__ == '__main__':
    main()

