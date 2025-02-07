import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations
from tqdm import tqdm, tqdm_notebook

def get_problem(path: str) -> dict:
    """Đọc dữ liệu bài toán từ file và lưu vào dictionary."""

    # Đọc nội dung file và tách từng dòng
    with open(path, 'r') as f:
        file = f.read().splitlines()

    problem_dict = {}  # Dictionary lưu dữ liệu bài toán
    section = None  # Biến kiểm soát phần nào của file đang đọc

    # Duyệt qua từng dòng trong file
    for i, row in enumerate(file):
        # 📌 Lấy số lượng xe tải và giá trị tối ưu từ dòng COMMENT
        if 'COMMENT' in row:
            _, comment = row.split(':', 1)
            problem_dict['n_trucks'] = int(comment.split('No of trucks:')[1].split(',')[0])
            problem_dict['optimal'] = int(comment.split('Optimal value:')[1].split(')')[0])

        # 📌 Lấy sức chứa của xe từ dòng CAPACITY
        elif 'CAPACITY' in row:
            problem_dict['capacity'] = int(row.split(':')[1].strip())

        # 📌 Khi gặp NODE_COORD_SECTION, bắt đầu đọc tọa độ điểm giao hàng
        elif 'NODE_COORD_SECTION' in row:
            section = 'NODE_COORD_SECTION'
            problem_dict['locations'] = []

        # 📌 Khi gặp DEMAND_SECTION, bắt đầu đọc nhu cầu đơn hàng
        elif 'DEMAND_SECTION' in row:
            section = 'DEMAND_SECTION'
            problem_dict['demands'] = []

        # 📌 Đọc tọa độ từng điểm giao hàng
        elif section == 'NODE_COORD_SECTION':
            node_id, x, y = row.split()
            problem_dict['locations'].append((float(x), float(y)))  # Lưu tọa độ dưới dạng tuple (x, y)

        # 📌 Đọc nhu cầu đơn hàng tại từng điểm
        elif section == 'DEMAND_SECTION':
            parts = row.split()
            if len(parts) >= 2:
                node_id, demand = parts[:2]
                demand = int(demand)
                if demand == 0:  # Nếu nhu cầu bằng 0, đây là kho hàng
                    problem_dict['depot_i'] = len(problem_dict['demands'])
                problem_dict['demands'].append(demand)

    # 📌 Kiểm tra số điểm tọa độ và số nhu cầu có khớp nhau không
    problem_dict['n_locations'] = len(problem_dict['locations'])
    assert problem_dict['n_locations'] == len(problem_dict['demands'])

    # 📌 Tính ma trận khoảng cách giữa các điểm
    locations = problem_dict['locations']
    n = problem_dict['n_locations']
    dists = np.zeros((n, n), dtype=np.float32)  # Khởi tạo ma trận khoảng cách

    for i in range(n):
        for j in range(i + 1, n):  # Chỉ tính khi j > i để tránh trùng
            x1, y1 = locations[i]
            x2, y2 = locations[j]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5  # Công thức Euclidean
            dists[i, j] = dists[j, i] = distance  # Ma trận đối xứng

    problem_dict['dists'] = dists  # Lưu ma trận khoảng cách vào dictionary

    return problem_dict


def write_solution(solution: list, cost: float, filename: str) -> None:
    depots = [i for i, node in enumerate(solution) if node == 0]
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / filename, 'w') as f:
        for i, d in enumerate(depots[:-1]):
            route = solution[depots[i]+1:depots[i+1]]
            f.write(f'Route #{i+1}: {" ".join(map(str, route))}\n')
        f.write(f'Cost: {cost:.2f}')