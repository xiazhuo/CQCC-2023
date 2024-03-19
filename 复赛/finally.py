#导入必须的库和函数
import numpy as np
import pyqpanda as pq
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.tensor import QTensor
from pyvqnet.nn.module import Module
from pyvqnet.utils import set_random_seed

set_random_seed(256)

# 待删
import matplotlib.pyplot as plt
import time

g_param_num_u4 = 3      # 构建U4门所需的参数个数（即Ry,Rz,Ry三个门的旋转角度）
g_param_num_double_gate = 4 * g_param_num_u4    # 构建二体门所需的参数个数（需要 CNOT门*2 + U4门*4）
g_layer_num = 0         # 基本块层数
g_qstate = None         # 目标量子态
g_umatrix = None        # 目标酉矩阵


def build_u4(qubit, params):
    """使用RZ,RY,RZ门搭建U4门"""
    circuit = pq.QCircuit()
    circuit.insert(pq.RZ(qubit, params[0]))
    circuit.insert(pq.RY(qubit, params[1]))
    circuit.insert(pq.RZ(qubit, params[2]))
    return circuit


def build_double_gate(qlist, params):
    """使用单体门和受控非门搭建二体门"""
    global g_param_num_u4

    circuit = pq.QCircuit()
    now = 0     # 当前已使用的参数个数
    circuit.insert(pq.CNOT(qlist[0], qlist[1]))
    circuit.insert(build_u4(qlist[0], params[now: now + g_param_num_u4]))
    now += g_param_num_u4
    circuit.insert(build_u4(qlist[1], params[now: now + g_param_num_u4]))
    now += g_param_num_u4

    circuit.insert(pq.CNOT(qlist[1], qlist[0]))
    circuit.insert(build_u4(qlist[0], params[now: now + g_param_num_u4]))
    now += g_param_num_u4
    circuit.insert(build_u4(qlist[1], params[now: now + g_param_num_u4]))
    return circuit


def build_layer(qlist, params, qubit_num):
    """在所有相邻的两个量子比特间叠加一层二体门，视为一层基本块"""
    global g_param_num_double_gate

    circuit = pq.QCircuit()
    for i in range(qubit_num - 1):
        double_gate = build_double_gate(
                    [qlist[i], qlist[i + 1]], 
                    params[i * g_param_num_double_gate: (i + 1) * g_param_num_double_gate])
        circuit.insert(double_gate)
    return circuit


def build_cir(qlist, params, layer_num, qubit_num):
    """构建通用量子线路"""
    global g_param_num_u4
    global g_param_num_double_gate
    param_num_layer = (qubit_num - 1) * g_param_num_double_gate     # 在所有相邻的两个量子比特间叠加一层二体门所需要的参数个数

    circuit = pq.QCircuit()
    now = 0     # 当前已使用的参数个数
    # 首先是初始化层，即对所有量子比特添加一层U4门
    for i in range(qubit_num):
        circuit.insert(build_u4(qlist[i], params[0 + now: g_param_num_u4 + now]))
        now += g_param_num_u4

    # 接着是layer_num层基本块
    for i in range(layer_num):
        layer = build_layer(
            qlist,
            params[0 + now: param_num_layer + now],
            qubit_num)
        circuit.insert(layer)
        now += param_num_layer
    return circuit


def qvc_circuit_qstate(input, params, qlist, clist, machine):
    """目标是量子态时的变分量子线路与损失函数定义"""
    global g_qstate
    global g_layer_num
    qubit_num = len(qlist)
    circuit = build_cir(qlist, params, g_layer_num, qubit_num)
    prog = pq.QProg()
    prog.insert(circuit)

    machine.directly_run(prog)
    qstate = np.array(machine.get_qstate())
    loss = np.linalg.norm(g_qstate.reshape(-1) - (np.exp(1j*params[-1]))*qstate) # 排除全局相位后计算制备态与目标态之间的欧式距离
    return np.log10(loss)   # 返回欧式距离的对数形式作为损失函数，避免梯度消失


def qvc_circuit_Umatrix(input, params, qlist, clist, machine):
    """目标是酉矩阵时的变分量子线路与损失函数定义"""
    global g_umatrix
    global g_layer_num
    qubit_num = len(qlist)
    circuit = build_cir(qlist, params, g_layer_num, qubit_num)
    prog = pq.QProg()
    prog.insert(circuit)

    machine.directly_run(prog)
    mat = np.array(pq.get_matrix(prog))
    loss = np.linalg.norm(g_umatrix.flatten() - np.exp(1j*params[-1])*mat)  # 排除全局相位后计算拟合的酉矩阵与目标酉矩阵之间的欧式距离
    return np.log10(loss)   # 返回欧式距离的对数形式作为损失函数，避免梯度消失


class Model(Module):
    """使用VQNet的优化算法进行模型训练"""
    def __init__(self, question, param_num, qubit_num):
        super(Model, self).__init__()
        global g_layer_num
        if question == "question1":
            self.pqc = QuantumLayer(qvc_circuit_qstate, param_num, "cpu", qubit_num)
            self.qubit_num = qubit_num
        elif question == "question2":
            self.pqc = QuantumLayer(qvc_circuit_Umatrix, param_num, "cpu", qubit_num)
            self.qubit_num = qubit_num

        self.best_loss = 0          # 记录训练过程中的最佳loss
        self.best_params = None     # 记录对应的最佳参数
        self.optimizer = adam.Adam(self.parameters(), lr=0.01)  # 使用自适应学习率的Adam优化器优化参数


    def forward(self):
        """定义前向传递逻辑"""
        input = QTensor([[None]]) # 必须要加才能过编译
        return self.pqc(input)
    
    def get_circuit(self, qlist):
        """返回参数训练后的最终电路"""
        return build_cir(qlist, self.best_params, g_layer_num, self.qubit_num)


def question1(quantum_state_vector, qlist):
    global g_param_num_u4
    global g_param_num_double_gate
    global g_qstate
    global g_layer_num
    g_qstate = quantum_state_vector
    qubit_num = len(qlist)
    g_layer_num = 3 if qubit_num == 3 else 4    # 制备3bits量子态需要三层基本块，4bits量子态需要四层基本块
    # 总参数量除了量子线路内部待训练的旋转门参数以外，还包括一个自适应的全局相位参数，用以排除全局相位的影响
    param_num = qubit_num * g_param_num_u4 + (qubit_num - 1) * g_param_num_double_gate * g_layer_num + 1

    model = Model("question1", param_num, qubit_num)
    model.train()
    epoch = 500     # 训练500次
    print("Start training")
    loss_arr = []

    for i in range(epoch):
        model.optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        model.optimizer._step()
        loss_arr.append(loss.to_numpy()[0])
        # 记录最佳的loss以及对应的参数
        if loss_arr[-1] < model.best_loss:
            model.best_loss = loss_arr[-1]
            model.best_params = model.pqc.m_para.to_numpy()[:-1]

        if i % 20 == 19:
            print(f"epoch {i} loss: {loss.to_numpy()[0]}")

    plt.plot(loss_arr)
    plt.show()

    return model.get_circuit(qlist)


def question2(unitary_matrix, qlist):
    global g_param_num_u4
    global g_param_num_double_gate
    global g_umatrix
    global g_layer_num
    g_umatrix = unitary_matrix
    qubit_num = len(qlist)
    g_layer_num = 6     # 制备QFT(3)酉矩阵需要6层基本块
    # 总参数量除了量子线路内部待训练的旋转门参数以外，还包括一个自适应的全局相位参数，用以排除全局相位的影响
    param_num = qubit_num * g_param_num_u4 + (qubit_num - 1) * g_param_num_double_gate * g_layer_num + 1

    model = Model("question2", param_num, qubit_num)
    model.train()
    epoch = 600     # 训练600次
    print("Start training")
    loss_arr = []

    for i in range(epoch):
        model.optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        model.optimizer._step()
        loss_arr.append(loss.to_numpy()[0])
        # 记录最佳的loss以及对应的参数
        if loss_arr[-1] < model.best_loss:
            model.best_loss = loss_arr[-1]
            model.best_params = model.pqc.m_para.to_numpy()[:-1]

        if i % 20 == 19:
            print(f"epoch {i} loss: {loss.to_numpy()[0]}")
            
    plt.plot(loss_arr)
    plt.show()

    return model.get_circuit(qlist)


# test model
def random_state_vector(qubit_num):
    """随机生成量子态"""
    state_vector = np.random.rand(2 ** qubit_num) + np.random.rand(2 ** qubit_num) * 1j
    state_vector /= np.linalg.norm(state_vector)
    return state_vector


def QFT_matrix(qubit_num):
    """QFT(3)"""
    qvm = pq.CPUQVM()
    qvm.init_qvm()
    qlist = qvm.qAlloc_many(qubit_num)
    prog = pq.QProg()

    prog << pq.QFT(qlist)

    qvm.prob_run_dict(prog, qlist, -1)
    mat = np.array(pq.get_matrix(prog)).reshape(2**qubit_num,2**qubit_num)
    return mat


def random_unitary_matrix(qubit_num):
    mat = np.random.rand(2 ** qubit_num, 2 ** qubit_num) + np.random.rand(2 ** qubit_num, 2 ** qubit_num) * 1j
    P, _, Q = np.linalg.svd(mat)
    mat = P.dot(Q)
    return mat

def print_question1(t_qnum):
    qvm = pq.CPUQVM()
    qvm.init_qvm()
    qlist = qvm.qAlloc_many(t_qnum)
    prog = pq.QProg()
    begin_time = time.time()
    # test_quantum_vector = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    # test_quantum_vector = 1 / np.sqrt(3) * np.array([0, 1, 1, 0, 1, 0, 0, 0])
    test_quantum_vector = random_state_vector(t_qnum)
    circuit = question1(test_quantum_vector, qlist)
    prog = pq.QProg()
    prog << circuit
    qvm.directly_run(prog)
    state = np.array(qvm.get_qstate())
    theta = np.inner(test_quantum_vector.conjugate(),state)
    f = np.linalg.norm(state/theta - test_quantum_vector)
    cnot_count = pq.count_qgate_num(prog, pq.GateType.CNOT_GATE)
    print('线路末态运行结果为{}，欧几里得距离为{}，使用的CNOT个数为{}'.format(state, f, cnot_count))
    tot_time = time.time() - begin_time
    print("运行总时间:\n", int(tot_time/60), ":", int(tot_time % 60))

def print_question2(t_qnum):
    qvm = pq.CPUQVM()
    qvm.init_qvm()
    qlist = qvm.qAlloc_many(t_qnum)
    prog = pq.QProg()
    begin_time = time.time()
    test_unitary_matrix = QFT_matrix(t_qnum)
    # test_unitary_matrix = random_unitary_matrix(t_qnum)
    circuit = question2(test_unitary_matrix, qlist)
    prog << circuit
    qvm.directly_run(prog)
    matrix = np.array(pq.get_matrix(prog)).reshape((2**t_qnum, 2**t_qnum))
    theta = np.trace(test_unitary_matrix.T.conjugate().dot(matrix)) / 2**t_qnum
    f = np.linalg.norm(matrix/theta - test_unitary_matrix, ord='fro')
    cnot_count = pq.count_qgate_num(prog, pq.GateType.CNOT_GATE)
    print('线路矩阵结果为{}，欧几里得距离为{}，使用的CNOT个数为{}'.format(matrix, f, cnot_count))
    tot_time = time.time() - begin_time
    print("运行总时间:\n", int(tot_time/60), ":", int(tot_time % 60))

if __name__ == '__main__':
    t_qnum = 3
    t_quest = 1

    if t_quest == 1:
        print_question1(t_qnum)
    else:
        print_question2(t_qnum)