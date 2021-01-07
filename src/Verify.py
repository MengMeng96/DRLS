from param import *

class Verify(object):
    def __init__(self, info, globalCycle=args.global_cycle, bufferCapacity=5):
        self.info = info
        self.globalCycle = globalCycle
        self.sendDelay = 1
        self.bufferCapacity = bufferCapacity

    def judge_conflict(self):
        flag = True
        for node in self.info.values():
            for port in node.values():
                for directions in port.values():
                    # print(directions)
                    intervals = []
                    for flow in directions.values():
                        # print(flow)
                        start = flow["offset"]
                        while start < self.globalCycle:
                            intervals.append([start, start + self.sendDelay])
                            start += flow["cycle"]
                    # print(intervals)
                    flag = flag and self.cover_count(intervals) < 2
        return flag

    def judge_buffer_overflow(self):
        flag = True
        for node in info.values():
            intervals = []
            for port in node.values():
                for directions in port.values():
                    for flow in directions.values():
                        # print(flow)
                        time = flow["offset"]
                        while time < self.globalCycle:
                            if directions == "send":
                                intervals.append([-1, time])
                            else:
                                intervals.append([time, -1])
                            time += flow["cycle"]
            flag = flag and self.cover_count(intervals) < self.bufferCapacity
        return flag

    def cover_count(self, intervals):
        # print(intervals)
        time_line = {}
        for interval in intervals:
            if interval[0] not in time_line:
                time_line[interval[0]] = 0
            if interval[1] not in time_line:
                time_line[interval[1]] = 0
            time_line[interval[0]] += 1
            time_line[interval[1]] -= 1
        if -1 in time_line:
            time_line[-1] = 0
        time_line = sorted(time_line.items())
        cur = 0
        cnt = 0
        # print(time_line)
        for item in time_line:
            cur += item[1]
            if cur > cnt:
                cnt = cur
        # print(cnt)
        return cnt


def main():
    verify = Verify(info)
    if not (verify.judge_conflict() and verify.judge_buffer_overflow()):
        print(False)


if __name__ == '__main__':
    main()