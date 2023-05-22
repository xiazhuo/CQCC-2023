import pyqpanda as pq
#导入必须的库和函数
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.tensor import QTensor
import numpy as np
from pyvqnet.nn.module import Module

# 待删
import matplotlib.pyplot as plt

qubit_num = 4
params_num_u4 = 3
params_num_double_gate = 12
layer_num = 3 if qubit_num == 4 else 2
param_num = 3 * qubit_num + params_num_double_gate * (qubit_num - 1) * layer_num + 1

def u4(qubit, params):
    """"用RZ,RY,RZ门构建U4门"""
    circuit = pq.QCircuit()
    circuit.insert(pq.RZ(qubit, params[0]))
    circuit.insert(pq.RY(qubit, params[1]))
    circuit.insert(pq.RZ(qubit, params[2]))
    return circuit

def double_gate(qlist, params):
    """双量子比特门"""
    circuit = pq.QCircuit()
    circuit.insert(pq.CNOT(qlist[0], qlist[1]))
    circuit.insert(u4(qlist[0], params[0:3]))
    circuit.insert(u4(qlist[1], params[3:6]))

    circuit.insert(pq.CNOT(qlist[1], qlist[0]))
    circuit.insert(u4(qlist[0], params[6:9]))
    circuit.insert(u4(qlist[1], params[9:12]))

    return circuit

def one_layer(qlist, params, qubit_num):
    """叠加一层双量子比特门"""
    circuit = pq.QCircuit()
    for i in range(qubit_num - 1):
        dgate = double_gate([qlist[i], qlist[i + 1]], params[i * params_num_double_gate: (i + 1) * params_num_double_gate])
        circuit.insert(dgate)
    return circuit

def multi_u_layer(qlist, params, layer_num, qubit_num):
    """搭建变分量子线路"""
    circuit = pq.QCircuit()
    now = 0
    for i in range(qubit_num):
        circuit.insert(u4(qlist[i], params[0 + now:3 + now]))
        now += 3
    par_num_layer = (qubit_num - 1) * 12
    for i in range(layer_num):
        layer = one_layer(
            qlist,
            params[0 + now:par_num_layer + now],
            qubit_num)
        circuit.insert(layer)
        now += par_num_layer
    return circuit

def qdrl_circuit_qstate(input, params, qlist, clist, machine):
    circuit = multi_u_layer(qlist, params, layer_num, qubit_num)
    prog = pq.QProg()
    prog.insert(circuit)

    # 目标是量子态
    machine.directly_run(prog)
    qstate = np.array(machine.get_qstate())
    loss = np.linalg.norm(quantum_state.reshape(-1) - np.e**(1j*params[-1])*qstate)

    return loss
    
def qdrl_circuit_Umatrix(input, params, qlist, clist, machine):
    circuit = multi_u_layer(qlist, params, layer_num, qubit_num)
    prog = pq.QProg()
    prog.insert(circuit)

    # 目标是酉矩阵
    machine.directly_run(prog)
    mat = np.array(pq.get_matrix(prog,True))
    loss = np.linalg.norm(u_matrix.flatten() - np.e**(1j*params[-1])*mat)

    return loss
    
class Model(Module):
    def __init__(self, question):
        super(Model, self).__init__()
        if question == "question1":
            self.pqc = QuantumLayer(qdrl_circuit_qstate, param_num, "cpu", qubit_num)
        elif question == "question2":
            self.pqc = QuantumLayer(qdrl_circuit_Umatrix, param_num, "cpu", qubit_num)
        self.optimizer = adam.Adam(self.parameters(), lr=0.01)

    def forward(self):
        input = QTensor([[None]]) # 必须要加才能过编译
        return self.pqc(input)
    
    def get_circuit(self, qlist):
        params = self.pqc.m_para.to_numpy()
        return multi_u_layer(qlist, params[:-1], layer_num, qubit_num)


def question1(quantum_state_vector, qlist):
    global qubit_num
    qubit_num = len(qlist)
    global quantum_state
    quantum_state = quantum_state_vector

    model = Model("question1")
    model.train()
    epoch = 500
    print("Start training")
    loss_arr = []

    for i in range(epoch):
        model.optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        model.optimizer._step()
        loss_arr.append(loss.to_numpy()[0])

        if i % 100 == 0:
            print(f"epoch {i} loss: {loss.to_numpy()[0]}")
            # print("param", model.pqc.m_para)

    plt.plot(loss_arr)
    plt.show()

    return model.get_circuit(qlist)

def question2(unitary_matrix, qlist):
    global u_matrix
    u_matrix = unitary_matrix
    global qubit_num
    qubit_num = len(qlist)
    
    model = Model("question2")
    model.train()
    epoch = 500
    print("Start training")
    loss_arr = []

    for i in range(epoch):
        model.optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        model.optimizer._step()
        loss_arr.append(loss.to_numpy()[0])

        if i % 100 == 0:
            print(f"epoch {i} loss: {loss.to_numpy()[0]}")
            # print("param", model.pqc.m_para)
            
    plt.plot(loss_arr)
    plt.show()

    return model.get_circuit(qlist)


# test model
def random_state_vector(qlist):
    """随机生成量子态"""
    qubit_num = len(qlist)
    state_vector = np.random.rand(2 ** qubit_num,1) + np.random.rand(2 ** qubit_num,1) * 1j
    state_vector /= np.linalg.norm(state_vector)
    return state_vector


qvm = pq.CPUQVM()
qvm.init_qvm()
qlist = qvm.qAlloc_many(3)
prog = pq.QProg()

# test_quantum_vector = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(16,1)
test_quantum_vector = 1 / np.sqrt(3) * np.array([0, 1, 1, 0, 1, 0, 0, 0]).reshape(8,1)
# quantum_state = random_state_vector(qlist)

circuit = question1(test_quantum_vector,qlist)
prog << circuit
print(prog)

result = qvm.prob_run_dict(prog, qlist, -1)
print(result)

stat = np.array(qvm.get_qstate())
print(stat)