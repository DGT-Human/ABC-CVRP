import random
import numpy as np
from copy import copy
from random import randint
import importlib as imp
from itertools import combinations
from tqdm import tqdm, tqdm_notebook
from utils import solution_handler, validate
from algorithm.base import Algorithm
# neighbor_operator = imp.reload(neighbor_operator)

class LocalSearch(Algorithm):
    def __init__(self, problem):
        """
        Khởi tạo thuật toán tìm kiếm cục bộ với một bài toán cụ thể.
        :param problem: Đối tượng bài toán cần tối ưu hóa.
        """
        self.problem = problem

    @property
    def name(self):
        """
        Trả về tên thuật toán.
        """
        return 'LocalSearch'

    # Thiết lập các tham số cho thuật toán
    def set_params(self, solution, n_iter, **params):
        """
        Thiết lập các tham số cho quá trình tìm kiếm cục bộ.
        :param solution: Giải pháp ban đầu.
        :param n_iter: Số lần lặp để tìm kiếm giải pháp tốt hơn.
        :param params: Các tham số khác có thể được sử dụng.
        """
        self.solution = copy(solution)  # Sao chép giải pháp ban đầu để không làm thay đổi bản gốc
        self.n_iter = n_iter  # Số lần lặp
        self.params = params  # Lưu trữ các tham số bổ sung

    # Hàm giải bài toán bằng thuật toán tìm kiếm cục bộ
    def solve(self, only_feasible=True, verbose=False):
        """
        Thực hiện tìm kiếm cục bộ để tối ưu hóa giải pháp.
        :param only_feasible: Nếu True, chỉ chấp nhận giải pháp hợp lệ.
        :param verbose: Nếu True, in ra thông tin trong quá trình tìm kiếm.
        :return: Giải pháp tối ưu và chi phí tương ứng.
        """
        # Tính toán chi phí của giải pháp ban đầu
        self.cur_cost = validate.compute_solution(self.problem, self.solution)
        if verbose:
            print('Start cost: {}'.format(self.cur_cost))

        feasible_saving = copy(self.solution)  # Lưu trữ giải pháp hợp lệ tốt nhất
        operator = NeighborOperator()  # Tạo đối tượng thực hiện các phép toán láng giềng

        # Thực hiện n_iter lần tìm kiếm cục bộ
        for _ in tqdm(range(self.n_iter), disable=(not verbose)):
            tmp_sol = operator.random_operator(self.solution)  # Tạo giải pháp láng giềng ngẫu nhiên
            cost = validate.compute_solution(self.problem, tmp_sol)  # Tính toán chi phí của giải pháp mới

            # Nếu chi phí mới nhỏ hơn hoặc bằng chi phí hiện tại, cập nhật giải pháp
            if self.cur_cost >= cost:
                self.cur_cost = cost
                self.solution = tmp_sol
                # Kiểm tra xem giải pháp có hợp lệ không
                if validate.check_solution(self.problem, self.solution):
                    feasible_saving = copy(self.solution)

        # Nếu chỉ chấp nhận giải pháp hợp lệ nhưng giải pháp cuối không hợp lệ,
        # thì quay lại giải pháp hợp lệ tốt nhất đã tìm thấy
        if ((only_feasible) and (not validate.check_solution(self.problem, self.solution))):
            self.solution = feasible_saving
            self.cur_cost = validate.compute_solution(self.problem, self.solution)

        return self.solution, self.cur_cost

class NeighborOperator:

    def __init__(self):
        self.operators = {
                          1: self.random_swap,
                          2: self.random_swap_sub,
                          3: self.random_insert,
                          4: self.random_insert_sub,
                          5: self.random_reversing,
                          6: self.random_swap_sub_reverse,
                          7: self.random_insert_sub_reverse
                          }
        pass


    def random_operator(self, _solution, patience=10, verbose=False):
        operators = list(self.operators)
        # p = [p / sum(operators) for p in operators]
        p = None
        rand_choice = np.random.choice(operators, p=p)
        random_oper = self.operators[rand_choice]
        return random_oper(_solution, patience=patience, verbose=verbose)

    @staticmethod
    def random_swap(_solution, patience=10, verbose=False):
        if verbose:
            print('random swap')
        solution = copy(_solution)
        sol_len  = len(solution)
        while patience > 0:
            i, j = random.sample(range(1, sol_len-1), 2)
            if i != j and solution[i] != 0 and solution[j] != 0:
                solution[i], solution[j] = copy(solution[j]), copy(solution[i])
                break
            patience -= 1
        return solution

    @staticmethod
    def random_swap_sub(_solution, patience=10, verbose=False):
        if verbose:
            print('random swap of subsequence')
        solution = copy(_solution)
        sol_len  = len(solution)
        while patience > 0:
            k = random.choice(range(2, 7))
            i, j = random.sample(range(1, sol_len-k-1), 2)
            if abs(i-j)>k and solution[i] != 0 and solution[j] != 0:
                if verbose:
                    print('Swap: ', solution[i:i+k], solution[j:j+k])
                solution[i:i+k], solution[j:j+k] = copy(solution[j:j+k]), copy(solution[i:i+k])

                # there shouldn`t be several depots in a row for example [0, 0,.. ]
                if validate.check_depots_sanity(solution):
                    break
            patience -= 1

        return solution

    @staticmethod
    def random_insert(_solution, patience=10, verbose=False):
        if verbose:
            print('random insertion')
        solution = copy(_solution)
        sol_len  = len(solution)
        while patience > 0:
            i, j = random.sample(range(1, sol_len-1), 2)
            if i != j and solution[i] != 0 and solution[j] != 0:
                i, j = copy(min(i, j)), copy(max(i, j))
                if solution[j+1] != 0 or solution[j-1]!=0:
                    solution[:i] = _solution[:i]
                    solution[i]  = _solution[j]
                    solution[i+1:j+1] = _solution[i:j]
                    solution[j+1:]    = _solution[j+1:]
                    break
            patience -= 1

        return solution

    @staticmethod
    def random_insert_sub(_solution, patience=10, verbose=False):
        if verbose:
            print('random insertion of subsequence')
        solution = copy(_solution)
        sol_len  = len(solution)
        while patience > 0:
            k = random.choice(range(2, 7))
            i, j = random.sample(range(1, sol_len-k-1), 2)
            if abs(i-j) <= k:
                continue
            i, j = copy(min(i, j)), copy(max(i, j))
            if verbose:
                print('Insert: ',_solution[j:j+k], 'to', i, i+k)
            solution[:i] = _solution[:i]
            solution[i:i+k]  = _solution[j:j+k]
            solution[i+k:j+k] = _solution[i:j]
            solution[j+k:]  = _solution[j+k:]

            # there shouldn`t be several depots in a row for example [0, 0,.. ]
            if validate.check_depots_sanity(solution):
                break
            patience -= 1

        return solution

    @staticmethod
    def random_reversing(_solution, patience=10, verbose=False):
        if verbose:
            print('random reversing a subsequence')
        solution = copy(_solution)
        sol_len  = len(solution)
        while patience > 0:
            i, j = random.sample(range(1, sol_len-1), 2)
            if i != j:
                i, j = copy(min(i, j)), copy(max(i, j))
                if  solution[j+1] != 0  and solution[i-1] != 0:
                    solution[i:j] = solution[i:j][::-1]
                    break
            patience -= 1
        return solution

    @staticmethod
    def random_swap_sub_reverse(_solution, patience=10, verbose=False):
        if verbose:
            print('random swap of reversed subsequence')
        solution = copy(_solution)
        sol_len  = len(solution)
        while patience > 0:
            k = random.choice(range(2, 7))
            i, j = random.sample(range(1, sol_len-k-1), 2)
            if abs(i-j)>k and solution[i] != 0 and solution[j] != 0:
                if verbose:
                    print('Swap: ', solution[i:i+k], solution[j:j+k])
                solution[i:i+k], solution[j:j+k] = copy(solution[j:j+k][::-1]), copy(solution[i:i+k][::-1])

                # there shouldn`t be several depots in a row for example [0, 0,.. ]
                if validate.check_depots_sanity(solution):
                    break
            patience -= 1

        return solution

    @staticmethod
    def random_insert_sub_reverse(_solution, patience=10, verbose=False):
        if verbose:
            print('random insertion of subsequence')
        solution = copy(_solution)
        sol_len  = len(solution)
        while patience > 0:
            k = random.choice(range(2, 7))
            i, j = random.sample(range(1, sol_len-k-1), 2)
            if abs(i-j) <= k:
                continue
            i, j = copy(min(i, j)), copy(max(i, j))
            if verbose:
                print('Insert: ',_solution[j:j+k], 'to', i, i+k)
            solution[:i] = _solution[:i]
            solution[i:i+k]  = _solution[j:j+k][::-1]
            solution[i+k:j+k] = _solution[i:j]
            solution[j+k:]  = _solution[j+k:]

            # there shouldn`t be several depots in a row for example [0, 0,.. ]
            if validate.check_depots_sanity(solution):
                break
            patience -= 1

        return solution
