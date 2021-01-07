#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse

parser = argparse.ArgumentParser(description='DAG_ML')

# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
parser.add_argument('--num_proc', type=int, default=1,
                    help='number of processors (default: 1)')
parser.add_argument('--num_exp', type=int, default=10,
                    help='number of experiments (default: 10)')
parser.add_argument('--query_type', type=str, default='tpch',
                    help='query type (default: tpch)')
parser.add_argument('--job_folder', type=str, default='./spark_env/tpch/',
                    help='job folder path (default: ./spark_env/tpch/)')
parser.add_argument('--result_folder', type=str, default='./results/',
                    help='Result folder path (default: ./results)')
parser.add_argument('--model_folder', type=str, default='./models/',
                    help='Model folder path (default: ./models)')

# -- Environment --
parser.add_argument('--exec_cap', type=int, default=100,
                    help='Number of total executors (default: 100)')
parser.add_argument('--num_init_dags', type=int, default=10,
                    help='Number of initial DAGs in system (default: 10)')
parser.add_argument('--num_stream_dags', type=int, default=100,
                    help='number of streaming DAGs (default: 100)')
parser.add_argument('--num_stream_dags_grow', type=float, default=0.2,
                    help='growth rate of number of streaming jobs  (default: 0.2)')
parser.add_argument('--num_stream_dags_max', type=float, default=500,
                    help='maximum number of number of streaming jobs (default: 500)')
parser.add_argument('--stream_interval', type=int, default=25000,
                    help='inter job arrival time in milliseconds (default: 25000)')
parser.add_argument('--executor_data_point', type=int,
                    default=[5, 10, 20, 40, 50, 60, 80, 100], nargs='+',
                    help='Number of executors used in data collection')
parser.add_argument('--reward_scale', type=float, default=100000.0,
                    help='scale the reward to some normal values (default: 100000.0)')
parser.add_argument('--alibaba', type=bool, default=False,
                    help='Use Alibaba dags (defaule: False)')
parser.add_argument('--var_num_dags', type=bool, default=False,
                    help='Vary number of dags in batch (default: False)')
parser.add_argument('--moving_delay', type=int, default=2000,
                    help='Moving delay (milliseconds) (default: 2000)')
parser.add_argument('--warmup_delay', type=int, default=1000,
                    help='Executor warming up delay (milliseconds) (default: 1000)')
parser.add_argument('--diff_reward_enabled', type=int, default=0,
                    help='Enable differential reward (default: 0)')
parser.add_argument('--new_dag_interval', type=int, default=10000,
                    help='new DAG arrival interval (default: 10000 milliseconds)')
parser.add_argument('--new_dag_interval_noise', type=int, default=1000,
                    help='new DAG arrival interval noise (default: 1000 milliseconds)')

# -- Multi resource environment --
parser.add_argument('--exec_group_num', type=int,
                    default=[50, 50], nargs='+',
                    help='Number of executors in each type group (default: [50, 50])')
parser.add_argument('--exec_cpus', type=float,
                    default=[1.0, 1.0], nargs='+',
                    help='Amount of CPU in each type group (default: [1.0, 1.0])')
parser.add_argument('--exec_mems', type=float,
                    default=[1.0, 0.5], nargs='+',
                    help='Amount of memory in each type group (default: [1.0, 0.5])')

# -- Evaluation --
parser.add_argument('--test_schemes', type=str,
                    default=['dynamic_partition'], nargs='+',
                    help='Schemes for testing the performance')

# -- TPC-H --
parser.add_argument('--tpch_size', type=str,
                    default=['2g','5g','10g','20g','50g','80g','100g'], nargs='+',
                    help='Numer of TPCH queries (default: [2g, 5g, 10g, 20g, 50g, 80g, 100g])')
parser.add_argument('--tpch_num', type=int, default=22,
                    help='Numer of TPCH queries (default: 22)')

# -- Visualization --
parser.add_argument('--canvs_visualization', type=int, default=1,
                    help='Enable canvs visualization (default: 1)')
parser.add_argument('--canvas_base', type=int, default=-10,
                    help='Canvas color scale bottom (default: -10)')

# -- Learning --
parser.add_argument('--policy_input_dim', type=int, default=8,
                    help='policy input dimensions to graph embedding (default: 5)')
parser.add_argument('--time_input_dim', type=int, default=1028,
                    help='time input dimensions to graph embedding (default: 3)')
parser.add_argument('--output_dim', type=int, default=8,
                    help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=8,
                    help='Maximum depth of root-leaf message passing (default: 8)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--ba_size', type=int, default=64,
                    help='Batch size (default: 64)')
parser.add_argument('--gamma', type=float, default=1,
                    help='discount factor (default: 1)')
parser.add_argument('--early_terminate', type=int, default=0,
                    help='Terminate the episode when stream is empty (default: 0)')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.0001,
                    help='Final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=1e-3,
                    help='Entropy weight decay rate (default: 1e-3)')
parser.add_argument('--log_file_name', type=str, default='log',
                    help='log file name (default: log)')
parser.add_argument('--master_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in master (default: 0)')
parser.add_argument('--worker_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in worker (default: 0)')
parser.add_argument('--master_gpu_fraction', type=float, default=0.5,
                    help='Fraction of memory master uses in GPU (default: 0.5)')
parser.add_argument('--worker_gpu_fraction', type=float, default=0.1,
                    help='Fraction of memory worker uses in GPU (default: 0.1)')
parser.add_argument('--average_reward_storage_size', type=int, default=100000,
                    help='Storage size for computing average reward (default: 100000)')
parser.add_argument('--reset_prob', type=float, default=0,
                    help='Probability for episode to reset (after x seconds) (default: 0)')
parser.add_argument('--reset_prob_decay', type=float, default=0,
                    help='Decay rate of reset probability (default: 0)')
parser.add_argument('--reset_prob_min', type=float, default=0,
                    help='Minimum of decay probability (default: 0)')
parser.add_argument('--num_agents', type=int, default=1,
                    help='Number of parallel agents (default: 16)')
parser.add_argument('--num_ep', type=int, default=10000000,
                    help='Number of training epochs (default: 10000000)')
parser.add_argument('--learn_obj', type=str, default='mean',
                    help='Learning objective (default: mean)')
parser.add_argument('--saved_model', type=str, default=None,
                    help='Path to the saved tf model (default: None)')
parser.add_argument('--check_interval', type=float, default=0.01,
                    help='interval for master to check gradient report (default: 10ms)')
parser.add_argument('--model_save_interval', type=int, default=1000,
                    help='Interval for saving Tensorflow model (default: 1000)')
parser.add_argument('--num_saved_models', type=int, default=1000,
                    help='Number of models to keep (default: 1000)')

# -- Spark interface --
parser.add_argument('--scheduler_type', type=str, default='dynamic_partition',
                    help='type of scheduling algorithm (default: dynamic_partition)')

parser.add_argument('--hid_dims', type=int, default=[16, 8], nargs='+',
                    help='hidden dimensions throughout graph embedding (default: [16, 8])')

parser.add_argument('--data_path', type=str, default='../resource/Random_NetWork/PCL/0/',
# parser.add_argument('--data_path', type=str, default='../resource/data/201/',
                    help='path of test data')
# 默认单位：ms
parser.add_argument('--tt_flow_cycles', type=int, default=[64, 128, 256, 512],
#parser.add_argument('--tt_flow_cycles', type=str, default=[512, 1024, 2048, 4096],
#parser.add_argument('--tt_flow_cycles', type=str, default=[256, 512, 1024, 2048],
                    help='tt cycles(ms)')
# 默认单位：ms
parser.add_argument('--global_cycle', type=int, default=1,#4096,#8196,#32768,
                    help='global cycle(ms)')
parser.add_argument('--link_failure_pos', type=int, default=300,#8196,#32768,
                    help='path of test data')

parser.add_argument('--slot_per_millisecond', type=int, default=4,#4096,#8196,#32768,
                    help='number of slots in 1024 millisecond')
parser.add_argument('--link_rate', type=int, default=10,#4096,#8196,#32768,
                    help='rate of the TT network(MBps)')
parser.add_argument('--network_width', type=int, default=5,#4096,#8196,#32768,
                    help='rate of the TT network(MBps)')


args = parser.parse_args()
args.global_cycle = int(args.global_cycle * args.slot_per_millisecond)


def gcd(a, b):
    if a < b:
        temp = b
        b = a
        a = temp
    remainder = a % b
    if remainder == 0:
        return b
    else:
        return gcd(remainder, b)


def lcm(a, b):
    remainder = gcd(a, b)
    return int(a * b / remainder)


for cycle in args.tt_flow_cycles:
    args.global_cycle = lcm(args.global_cycle, cycle)
