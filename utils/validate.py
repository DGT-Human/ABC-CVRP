import random
import numpy as np
import pandas as pd
from copy import copy
from datetime import datetime
from tqdm import tqdm, tqdm_notebook


def compute_solution(problem, solution) -> np.float32:
    """
    Tính tổng chi phí của lời giải dựa trên ma trận khoảng cách.

    Công thức: cost = Sum Dij * Xij
    - Dij: Khoảng cách từ điểm i đến điểm j.
    - Xij: 1 nếu có chuyến đi từ i đến j, 0 nếu không.

    :param problem: Dictionary chứa thông tin về bài toán CVRP.
    :param solution: Danh sách các điểm theo thứ tự ghé thăm.
    :return: Tổng chi phí của lộ trình.
    """
    n = problem['n_locations']
    x = np.zeros((n, n), dtype=np.int32)  # Ma trận kề biểu diễn lộ trình

    # Duyệt qua từng cặp điểm liên tiếp trong lời giải và đánh dấu trong ma trận kề
    for i, loc in enumerate(solution[:-1]):
        x[solution[i], solution[i + 1]] = 1

    # Tính tổng chi phí dựa trên ma trận khoảng cách
    cost = problem['dists'][x == 1].sum()
    return cost


def check_solution(problem, solution, x=None, verbose=False) -> bool:
    """
    Kiểm tra xem lời giải có hợp lệ không theo các điều kiện sau:
    1. Độ dài lời giải phải bằng tổng số xe + số địa điểm.
    2. Điểm đầu và cuối của lộ trình phải là depot.
    3. Không có hai depot liên tiếp.
    4. Mọi địa điểm phải xuất hiện trong lời giải.
    5. Đảm bảo số xe khởi hành và quay lại depot đúng với `n_trucks`.
    6. Mỗi địa điểm chỉ được ghé thăm một lần.
    7. Đảm bảo tổng tải trọng mỗi tuyến không vượt quá tải trọng xe.

    :return: True nếu hợp lệ, False nếu không hợp lệ.
    """
    # Kiểm tra 1: Độ dài lời giải phải đúng
    sol_len = len(solution)
    plan_len = problem['n_trucks'] + problem['n_locations']
    if not sol_len == plan_len:
        if verbose:
            print('Solution len {} but should be {}'.format(sol_len, plan_len))
        return False  # Lỗi nếu độ dài sai

    # Kiểm tra 2: Depot phải ở đầu và cuối hành trình
    depots = list(filter(lambda i: solution[i] == 0, range(sol_len)))
    if depots[0] != 0 or depots[-1] != sol_len - 1:
        if verbose:
            print('The end and the start of the solution should be depots')
            print(depots)
        return False

    # Kiểm tra 3: Không có hai depot liên tiếp
    for i in range(len(depots) - 1):
        if depots[i + 1] - depots[i] <= 1:
            if verbose:
                print('Several depots in a row: {}'.format(depots))
            return False

    # Tạo ma trận kề nếu chưa có
    if not isinstance(x, np.ndarray):
        n = problem['n_locations']
        x = np.zeros((n, n), dtype=np.int32)
        for i, loc in enumerate(solution[:-1]):
            x[solution[i], solution[i + 1]] = 1

    # Kiểm tra 4: Mọi địa điểm phải xuất hiện ít nhất một lần
    if len(np.unique(solution)) != problem['n_locations']:
        if verbose:
            print('Failed locations sanity check')
            for i in range(problem['n_locations']):
                if i not in solution:
                    print('Missing: {} location'.format(i))
                    break
        return False

    # Kiểm tra 5: Số xe sử dụng phải đúng với `n_trucks`
    if not check_M_criteria(problem, solution, x=x, verbose=verbose):
        return False

    # Kiểm tra 6: Mỗi địa điểm phải được ghé thăm đúng một lần
    if not check_One_criteria(problem, solution, x=x, verbose=verbose):
        return False

    # Kiểm tra 7: Tổng tải trọng của mỗi tuyến không vượt quá tải trọng xe
    if not check_capacity_criteria(problem, solution, verbose=verbose):
        return False

    return True


def check_depots_sanity(solution):
    """ Kiểm tra xem có depot nào xuất hiện liên tiếp không (không hợp lệ). """
    sol_len = len(solution)
    depots = list(filter(lambda i: solution[i] == 0, range(sol_len)))
    for i in range(len(depots) - 1):
        if abs(depots[i + 1] - depots[i]) <= 1:
            return False
    return True


def check_One_criteria(problem, solution, x=None, verbose=False) -> bool:
    """
    Đảm bảo mỗi địa điểm được ghé đúng một lần (ngoại trừ depot).
    """
    if not ((x.sum(axis=1)[1:].sum() == problem['n_locations'] - 1) and
            (x.sum(axis=0)[1:].sum() == problem['n_locations'] - 1)):
        if verbose:
            print('Sum Xij for j = ', x.sum(axis=1)[1:])
            print('Sum Xij for j = ', x.sum(axis=0)[1:])
        return False
    return True


def check_M_criteria(problem, solution, x=None, verbose=False) -> bool:
    """
    Kiểm tra xem số lượng xe xuất phát từ depot và quay lại có đúng không.
    """
    if not isinstance(x, np.ndarray):
        n = problem['n_locations']
        x = np.zeros((n, n), dtype=np.int32)
        for i, loc in enumerate(solution[:-1]):
            x[solution[i], solution[i + 1]] = 1

    if not ((x[0, :].sum() == problem['n_trucks']) and
            (x[:, 0].sum() == problem['n_trucks'])):
        if verbose:
            print('n_trucks =', problem['n_trucks'])
            print('Sum Xi0 = ', x[:, 0].sum())
            print('Sum X0j = ', x[0, :].sum())
            print(solution)
        return False
    return True


def check_capacity_criteria(problem, solution, verbose=False) -> bool:
    """
    Kiểm tra xem tổng tải trọng mỗi tuyến có vượt quá tải trọng xe không.
    """
    capacity = problem['capacity']
    routes_demand = get_routes_demand(problem, solution)
    for route_demand in routes_demand:
        if route_demand > capacity:
            if verbose:
                print('Route demand {} exceeds capacity {}'.format(route_demand, capacity))
                print('Route ', routes_demand)
            return False
    return True


def get_routes(solution):
    """ Tách lời giải thành các tuyến đường con dựa trên vị trí depot. """
    sol_len = len(solution)
    depots = list(filter(lambda i: solution[i] == 0, range(sol_len)))
    routes = []
    for i, d in enumerate(depots[:-1]):
        route = solution[depots[i] + 1: depots[i + 1]]
        routes.append(route)
    return routes


def get_routes_demand(problem, _solution):
    """
    Tính toán tổng nhu cầu hàng hóa của từng tuyến đường.
    """
    solution = copy(_solution)
    sol_len = len(solution)
    demands = problem['demands']
    depots = list(filter(lambda i: solution[i] == 0, range(sol_len)))
    routes = []
    for i, d in enumerate(depots[:-1]):
        route = solution[depots[i] + 1: depots[i + 1]]
        route_demand = np.sum([demands[place] for place in route])
        routes.append(route_demand)
    return routes
