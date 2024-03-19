import sys
from answer.answer import question1, question2
import pyqpanda as pq
import numpy as np


def check_para_question_1(para):
    # 判断para是否是列表或np的array格式
    if isinstance(para, list) or isinstance(para, np.ndarray):
        # 判断para的长度是否是2的整数幂
        if len(para) > 0 and (len(para) & (len(para) - 1)) == 0:
            try:
                para = np.array(para)
                para_mod = np.abs(para)
                # 判断para的取模之后平方求和是否为1
                if np.isclose(np.sum(para_mod ** 2), 1):
                    return para
                else:
                    print('输入的用例必须是量子态向量，满足模方和为1')
            except:
                raise ValueError('输入的参数必须是数')
        else:
            raise ValueError('输入的参数长度必须是2的整数幂')
    else:
        raise ValueError('输入的参数必须是列表或np的array格式')


def is_unitary(para):
    if isinstance(para, (list, np.ndarray)):
        para = np.array(para)
        rows, cols = para.shape
        if rows == cols and (rows & (rows - 1) == 0) and np.issubdtype(para.dtype, np.number):
            if np.allclose(np.eye(rows), para @ para.conj().T):
                return para
            else:
                raise ValueError('输入的用例是幺正矩阵')
        else:
            raise ValueError('输入的用例是幺正矩阵')
    else:
        raise ValueError('输入的用例是幺正矩阵')


if __name__ == '__main__':
    args = sys.argv
    # question1的标准用例是problem_a problem_b problem_d，question2的标准用例是problem_c 。
    # 另外，question1还可以是长度为2^n、模方和为1的数组；
    # question2还可以是大小为(2^n, 2^n)的、幺正的数组。
    # 如果有问题 应该报错

    def print_question1(dataset):
        qvm = pq.CPUQVM()
        qvm.init_qvm()
        qubit_num = int(np.log2(len(dataset)))
        qlist = qvm.qAlloc_many(qubit_num)
        cir = question1(dataset, qlist)
        prog = pq.QProg()
        prog << cir
        qvm.directly_run(prog)
        state = np.array(qvm.get_qstate())
        f = np.linalg.norm(state - dataset)
        cnot_count = pq.count_qgate_num(prog, pq.GateType.CNOT_GATE)
        print('线路末态运行结果为{}，欧几里得距离为{}，使用的CNOT个数为{}'.format(state, f, cnot_count))

    def print_question2(dataset):
        qvm = pq.CPUQVM()
        qvm.init_qvm()
        shape = int(np.log2(dataset.shape[0]))
        qlist = qvm.qAlloc_many(shape)
        cir = question2(dataset, qlist)
        prog = pq.QProg()
        prog << cir
        qvm.directly_run(prog)
        matrix = np.array(pq.get_matrix(prog)).reshape((2**shape, 2**shape))
        f = np.linalg.norm(matrix - dataset, ord='fro')
        cnot_count = pq.count_qgate_num(prog, pq.GateType.CNOT_GATE)
        print('线路矩阵结果为{}，欧几里得距离为{}，使用的CNOT个数为{}'.format(matrix, f, cnot_count))

    if args[1] == 'question1':
        if args[2] == 'problem_a':
            problem_a = np.zeros(2 ** 4)
            problem_a[0] = 1
            problem_a[-1] = 1
            dataset = problem_a/np.sqrt(2)

        elif args[2] == 'problem_b':
            problem_b = np.zeros(2 ** 3)
            problem_b[1] = 1
            problem_b[2] = 1
            problem_b[4] = 1
            dataset = problem_b/np.sqrt(3)

        elif args[2] == 'problem_d':
            amplitude = np.random.random(8)
            amplitude = amplitude / (np.sum(amplitude ** 2) ** 0.5)
            angle = np.random.random(8) * 2 * np.pi
            problem_d = amplitude * np.exp(1j * angle)
            dataset = problem_d
        else:
            try:
                dataset = check_para_question_1(args[2])
            except:
                print('请输入正确的用例')
        print(print_question1(dataset))

    elif args[1] == 'question2':
        if args[2] == 'problem_c':
            dataset = np.array([[3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j,
                       3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j,
                       3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j,
                       3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j],
                      [3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j,
                       3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j,
                       -3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j,
                       -3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j],
                      [3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j,
                       -3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j,
                       2.16489014e-17 + 0.35355339j, 2.16489014e-17 + 0.35355339j,
                       -2.16489014e-17 - 0.35355339j, -2.16489014e-17 - 0.35355339j],
                      [3.53553391e-01 + 0.j, 3.53553391e-01 + 0.j,
                       -3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j,
                       -2.16489014e-17 - 0.35355339j, -2.16489014e-17 - 0.35355339j,
                       2.16489014e-17 + 0.35355339j, 2.16489014e-17 + 0.35355339j],
                      [3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j,
                       2.16489014e-17 + 0.35355339j, -2.16489014e-17 - 0.35355339j,
                       2.50000000e-01 + 0.25j, -2.50000000e-01 - 0.25j,
                       -2.50000000e-01 + 0.25j, 2.50000000e-01 - 0.25j],
                      [3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j,
                       2.16489014e-17 + 0.35355339j, -2.16489014e-17 - 0.35355339j,
                       -2.50000000e-01 - 0.25j, 2.50000000e-01 + 0.25j,
                       2.50000000e-01 - 0.25j, -2.50000000e-01 + 0.25j],
                      [3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j,
                       -2.16489014e-17 - 0.35355339j, 2.16489014e-17 + 0.35355339j,
                       -2.50000000e-01 + 0.25j, 2.50000000e-01 - 0.25j,
                       2.50000000e-01 + 0.25j, -2.50000000e-01 - 0.25j],
                      [3.53553391e-01 + 0.j, -3.53553391e-01 + 0.j,
                       -2.16489014e-17 - 0.35355339j, 2.16489014e-17 + 0.35355339j,
                       2.50000000e-01 - 0.25j, -2.50000000e-01 + 0.25j,
                       -2.50000000e-01 - 0.25j, 2.50000000e-01 + 0.25j]])
        else:
            try:
                dataset = is_unitary(args[2])
            except:
                print('请输入正确的用例')
        print(print_question2(dataset))