import sys
# 系统路径进入到上层目录，可以引用上层目录的库
sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")

import tensorflow.contrib.layers as tl
import time
from gcn import GraphCNN
from agent import Agent
from Environment import *
from Verify import *

from param import *
from tf_op import *
from utils import *


class ActorAgent(Agent):
    def __init__(self, sess, policy_input_dim, time_input_dim, hid_dims, output_dim,
                 max_depth, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='actor_agent'):

        Agent.__init__(self)

        self.sess = sess
        self.policy_input_dim = policy_input_dim
        self.time_input_dim = time_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope
        self.main_model = ""
        self.reschedule_cnt = -1
        self.reschedule_start_time = -1
        self.reschedule_end_time = -1

        # node input dimension: [total_num_nodes, num_features]
        self.policy_inputs = tf.placeholder(tf.float32, [None, args.policy_input_dim])

        # 8维
        self.gcn_policy = GraphCNN(
            self.policy_inputs, self.policy_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, "policy")

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        # gsn只有一个输出，是整图编码
        self.policy_act_probs = self.actor_network(
            self.policy_inputs, self.gcn_policy.outputs, self.act_fn)

        # draw action based on the probability (from OpenAI baselines)
        # node_acts [batch_size, 1]
        logits = tf.log(self.policy_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        # 这处对 noise 连用两个log好奇怪
        self.policy_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.edge_selected_vec = tf.placeholder(tf.float32, [None, None])
        # Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])

        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.placeholder(tf.float32, [1])

        # select node action probability
        # node_act_probs：一个全连接网络
        # node_act_vec：一个占位符
        # reduction_indices指定reduce_sum求和的轴维度
        # keep_dims求和后是否保持原有维度
        # node_act_vec是一个表示节点选取的向量，除了选出来的节点为1外，其他位置都为0
        self.selected_edge_prob = tf.reduce_sum(tf.multiply(self.policy_act_probs, self.edge_selected_vec))

        # actor loss due to advantge (negated)
        # reduce_sum：计算张量某一维的和
        # multiply：矩阵各对应位置元素相乘
        # self.eps：一个常量，感觉是用来防止和为0的
        # self.adv：一个占位符
        self.edge_act_loss = tf.multiply(tf.log(self.selected_edge_prob + 1e-6), -self.adv)
        # prob on each job
        # self.prob_each_job = tf.reshape(
        #     tf.sparse_tensor_dense_matmul(self.gsn.summ_mats[0],
        #         tf.reshape(self.node_act_probs, [-1, 1])),
        #         [tf.shape(self.node_act_probs)[0], -1])

        # define combined loss
        # self.act_loss = self.adv_loss

        # get training parameters
        self.edge_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy")

        # actor gradients
        self.edge_act_gradients = tf.gradients(self.edge_act_loss, self.edge_params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        # self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_edge_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.edge_act_gradients, self.edge_params))

        self.env = None

        # network paramter saver
        self.saver_policy = tf.train.Saver(self.edge_params, max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

    # gsn只有一个量，
    def actor_network(self, policy_inputs, gcn_policy_outputs, act_fn):

        # takes output from graph embedding and raw_input from environment
        batch_size = 1 # tf.shape(policy_inputs)[0]

        # (1) reshape node inputs to batch format
        policy_inputs_reshape = tf.reshape(
            policy_inputs, [batch_size, -1, args.policy_input_dim])

        # (4) reshape gcn_outputs to batch format
        gcn_policy_outputs_reshape = tf.reshape(
            gcn_policy_outputs, [batch_size, -1, self.output_dim])

        # (4) actor neural network
        with tf.variable_scope("policy"):
            # -- part A, the distribution over nodes --
            policy_input = tf.concat([policy_inputs_reshape, gcn_policy_outputs_reshape], axis=2)
            # print('merge_node', merge_node.shape)
            # 节点网络结构
            policy_hid_0 = tl.fully_connected(policy_input, 32, activation_fn=act_fn)
            policy_hid_1 = tl.fully_connected(policy_hid_0, 16, activation_fn=act_fn)
            policy_hid_2 = tl.fully_connected(policy_hid_1, 8, activation_fn=act_fn)
            policy_outputs = tl.fully_connected(policy_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, total_num_nodes)
            policy_outputs = tf.reshape(policy_outputs, [batch_size, -1])

            # 使用绝对值处理DNN输出
            # policy_abs = tf.abs(policy_outputs)
            # policy_max = tf.reduce_max(policy_abs, keep_dims=True, axis=-1)
            # policy_outputs = tf.divide(policy_outputs, policy_max)
            # policy_outputs = tf.nn.softmax(policy_outputs, dim=-1)

            # 使用减去最小值处理网络输出
            policy_min = tf.reduce_min(policy_outputs, keep_dims=True, axis=-1)
            policy_outputs = tf.subtract(policy_outputs, policy_min)
            policy_max = tf.reduce_max(policy_outputs, keep_dims=True, axis=-1)
            policy_outputs = tf.divide(policy_outputs, policy_max)
            policy_outputs = tf.nn.softmax(policy_outputs, dim=-1)

            return policy_outputs

    def apply_edge_gradients(self, gradients, lr_rate):
        # print("apply_edge_gradients")
        self.sess.run(self.apply_edge_grads, feed_dict={
            i: d for i, d in zip(
                self.edge_act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def invoke_model(self, manual):
        # implement this module here for training
        # (to pick up state and action to record)
        valid_edges, policy_inputs, time_inputs, cycle, time_offset, flow_length, max_delay = self.env.translate_data_to_inputs()
        if manual and len(valid_edges) == 0:
            #print("manual error!")
            return policy_inputs, time_inputs, -1, -2, -1, -1, -1, False
        # invoke learning model
        # 使用深度网络得到节点和任务决策
        # 每一条合法边都需要被判断一次，每一次的可达性矩阵是不同的，这能够产生针对当前边的gcn
        res = []
        # print(len(policy_inputs))
        # print(len(self.env.graph.reachable_edge_matrix), len(self.env.graph.reachable_edge_matrix[0]),
        # len(self.env.graph.reachable_edge_matrix[0][0]))
        edge_act_probs, edge_acts, gcn = self.sess.run(
            [self.policy_act_probs, self.policy_acts, self.gcn_policy.outputs],
            feed_dict={i: d for i, d in zip(
                [self.policy_inputs] + [self.gcn_policy.reachable_edge_matrix],
                [policy_inputs] + [self.env.graph.reachable_edge_matrix])
                       })
        # print("gcn \n", gcn)
        # print("policy inputs:\n", policy_inputs)
        # print("policy ", edge_act_probs)
        scope = range(len(policy_inputs))
        if manual:
            scope = valid_edges
        for edge_id in scope:
            res.append([edge_id, edge_act_probs[0, edge_id]])
        res = sorted(res, key=lambda edge_info: edge_info[1])
        # print("edge network: ", res)
        # 选边
        edge_info = res[-1]
        edge = self.env.graph.edges[edge_info[0]]

        # 计算策略网络梯度梯度：给出选中的边的mask
        edge_selected_mask = np.zeros([1, len(self.env.graph.edges)])
        edge_selected_mask[0, edge.id] = 1

        # 时延应该 从发出时间算，而不是从0时刻开始算
        start_time = -1
        offset = -1
        if len(self.env.tt_flow_time_record) > 1:
            start_time = self.env.tt_flow_time_record[-1] - self.env.tt_flow_time_record[1]
            offset = self.env.tt_flow_time_record[-1] % args.global_cycle
        time_slot, LD_score = edge.find_time_slot(start_time, offset, cycle, flow_length, max_delay)
        return policy_inputs, time_inputs, edge, time_slot, edge_selected_mask, cycle, LD_score, True

    def compute_edge_gradient(self, policy_inputs, gcn_policy_masks, edge, edge_selected_vec, adv):
        gradients, loss = self.sess.run([self.edge_act_gradients, self.edge_act_loss],
                                        feed_dict={i: d for i, d in zip(
                [self.policy_inputs] + [self.gcn_policy.reachable_edge_matrix] + [self.edge_selected_vec] + [self.adv], \
                [policy_inputs] + [gcn_policy_masks] + [edge_selected_vec] + [adv])
                        })
        # print("edge loss", edge_probs, loss)
        return gradients, loss


def discount(rewards, gamma):
    res = [i for i in rewards]
    total = len(rewards)
    for i in reversed(range(len(rewards) - 1)):
        res[i] += gamma * res[-1] * (total - (len(rewards) - i - 1)) / total
    return res


def test(actor_agent):
    actor_agent.env = Environment()
    start_time = time.time()
    done = 0
    flow_number = 0
    per_flow_cnt_total = 0
    per_flow_cnt_valid = 0
    time_record = {}
    manual = True
    try_model_id = 0
    models = ["../models/A380_Model/A" + str(model_count) + "/policy/" for model_count in range(34)]
    while True:
        while done == 0:
            per_flow_cnt_total += 1
            per_flow_cnt_valid += 1
            policy_inputs, time_inputs, edge, time_slot, edge_selected_mask, cycle, LD_score, flag = actor_agent.invoke_model(manual)
            reward, done, reason = actor_agent.env.step(edge, time_slot, LD_score)
            # print(edge.id, time_slot, actor_agent.env.time, actor_agent.env.tt_flow_deadline)
        cur_time = time.time()
        if done == -1:
            if try_model_id < len(models):
                actor_agent.env.roll_back(1)
                # print(actor_agent.env.edge_usage())
                # print(models[len(models) - 1 - try_model_id])
                actor_agent.saver_policy.restore(actor_agent.sess, models[len(models) - 1 - try_model_id])
                try_model_id += 1
                actor_agent.env.enforce_next_query()
                done = 0
            else:
                break
        elif done == 1: # 继续调度下一条流
            delay = actor_agent.env.tt_flow_time_record[-1] - actor_agent.env.tt_flow_time_record[0]
            usage = actor_agent.env.edge_usage()
            # print("TT_flow", flow_number, "cycle", cycle, "usage", usage,  "use time", cur_time - start_time, "delay", delay, "reward", reward)
            time_record[flow_number] = [flow_number, cycle, per_flow_cnt_total, per_flow_cnt_valid, cur_time - start_time, delay]
            # 删除边之后重新调度受影响的流
            if flow_number == args.link_failure_pos:
                actor_agent.reschedule_start_time = time.time()
                edge = actor_agent.env.graph.edges[0]
                actor_agent.reschedule_cnt = len(actor_agent.env.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)])
                # print(actor_agent.env.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)], actor_agent.reschedule_start_time)
                actor_agent.env.delete_edge((edge.start_node.id, edge.end_node.id))
            if actor_agent.env.reschedule == 2:
                actor_agent.reschedule_end_time = time.time()
                # print(actor_agent.reschedule_end_time)
            # 修改结束
            actor_agent.env.enforce_next_query()  # 调度下一条流
            start_time = time.time()
            flow_number += 1
            per_flow_cnt_total = 0
            per_flow_cnt_valid = 0
            done = 0
            if try_model_id != 0:
                try_model_id = 0
                actor_agent.saver_policy.restore(actor_agent.sess, actor_agent.main_model)
        if actor_agent.env.tt_query_id == 59999:
            # actor_agent.env.schedule.show()
            break
    edge_usage = actor_agent.env.edge_usage()

    verify = Verify(actor_agent.env.schedule.sche)
    print(actor_agent.reschedule_end_time, actor_agent.reschedule_start_time)
    return actor_agent.env.tt_query_id, edge_usage, verify.judge_conflict(), \
           actor_agent.reschedule_end_time - actor_agent.reschedule_start_time, actor_agent.reschedule_cnt


def write_result_init():
    result_file_path = "Result/temp.xls"
    result_sheet = "DRLS_A380"
    value_title = [["Data", "TS Number", "Link Usage", "Valid Test", "Total Time", "Time Per Flow",
                    "reschedule time", "reschedule count"]]
    write_excel_xls(result_file_path, result_sheet, value_title)


def write_result(result):
    result_file_path = "Result/temp.xls"
    write_excel_xls_append(result_file_path, result)


def main():
    np.random.seed()
    tf.set_random_seed(0)

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))
    sess = tf.Session(config=config)
    actor_agent = ActorAgent(
        sess, args.policy_input_dim, 1028,
        args.hid_dims, args.output_dim, args.max_depth)

    # 加载已经训练好的模型，参数是模型位置
    actor_agent.main_model = "../models/A380_Model/A33/policy/"
    # actor_agent.main_model = "../models/Ladder_Model/L18/policy/"
    actor_agent.saver_policy.restore(actor_agent.sess, actor_agent.main_model)
    datasets = []
    for i in range(1, 100):
        datasets.append(f"../resource/A380_NetWork/{i}/")
    cur_time = time.time()
    # 创建结果文件
    write_result_init()
    for dataset in datasets:
        args.data_path = dataset
        num, usage, flag, reschedule_time, reschedule_cnt = test(actor_agent)
        result = [args.data_path, num, usage, flag, time.time() - cur_time, 0, reschedule_time, reschedule_cnt]
        # 将结果写入文件
        write_result([result])
        print(result)
        cur_time = time.time()


if __name__ == '__main__':
    main()
