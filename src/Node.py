from param import *

class Node:
    def __init__(self, index, capacity):
        self.id = index
        self.buffer_capacity = capacity
        self.is_source_node = 0
        self.is_destination_node = 0

        # 每个时隙的使用情况，只需要记录一个周期的
        self.buffer_avaiable = [self.buffer_capacity for _ in range(args.global_cycle)]

    def set_source_node(self):
        self.is_source_node = 1

    def set_destination_node(self):
        self.is_destination_node = 1

    def unset_source_node(self):
        self.is_source_node = 0

    def unset_destination_node(self):
        self.is_destination_node = 0

    def check_buffer(self, start, cycle):
        for pos in range(args.global_cycle):
            if pos % cycle == start:
                if self.buffer_avaiable[pos] - 1 < 0:
                    return False
        return True

    def occupy_buffer(self, start, cycle):
        for pos in range(args.global_cycle):
            if pos % cycle == start:
                self.buffer_avaiable[pos] -= 1

    def check_buffers(self, start, end, cycle):
        for pos in range(args.global_cycle):
            offset = pos % cycle
            if (start < end and start < offset <= end) or \
                    (start > end and (offset <= end or start < offset)):
                if self.buffer_avaiable[pos] - 1 < 0:
                    return False
        return True

    def occupy_buffers(self, start, end, cycle):
        # 不占用开始时隙的缓存，因为开始时隙的缓存是在确定上一条边的时候占用的
        for pos in range(args.global_cycle):
            offset = pos % cycle
            if (start < end and start < offset <= end) or \
                    (start > end and (offset <= end or start < offset)):
                self.buffer_avaiable[pos] -= 1

    def reset(self):
        self.is_source_node = 0
        self.is_destination_node = 0
        self.buffer_avaiable = [self.buffer_capacity for _ in range(args.global_cycle)]

    def show(self, cycle):
        for i in range(args.global_cycle):
            print(self.buffer_avaiable[i], end=' ')
            if i % cycle == cycle - 1:
                print()

def main():
    node = Node(0, 1)
    node.check_buffers(61, 3, 64)
    node.occupy_buffers(3, 61, 64)
    for i in range(args.global_cycle):
        print(node.buffer_avaiable[i], end=' ')
        if i % 64 == 63:
            print()


if __name__ == '__main__':
    main()