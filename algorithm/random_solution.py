import random
import numpy as np
from utils import validate


def generate_solution(problem,
                      patience=50, # số lần thử lại
                      verbose=False) -> np.ndarray:
    dists   = problem['dists']
    demands = problem['demands']

    for itr in range(patience):

        i_loc   = [i for i in range(1, problem['n_locations'])]
        routes  = [[0] for _ in range(problem['n_trucks'])]

        for i in range(len(i_loc)):
            route_dists = []
            random_loc  = random.choice(i_loc)
            for route in routes:
                dist_to_loc  = dists[route[-1]][random_loc]
                route_demand = sum([demands[i] for i in route]) + demands[random_loc]
                demand_penalty = max(0, route_demand - problem['capacity'])
                P = 10
                # if  route_demand > problem['capacity']:
                #     coef = MAXIMUM_PENALTY
                # else:
                coef = len(route) * 2 + dist_to_loc + demand_penalty * P   # tính xác suất chọn mỗi tuyến đường
                # (số lượng điểm * 2 + khoảng cách điểm cuối với điểm muốn thêm + vượt quá tài trọng * 10)
                route_dists.append(coef)

            routes[np.argmin(route_dists)].append(random_loc) # chọn route (tuyến) có coef nhỏ nhất
            i_loc.remove(random_loc)

        solution = [loc for route in routes for loc in route] # tạo thành 1 mảng từ nhiều xe
        solution.append(0) # gán 0 ở cuối mảng trên để đúng ràng buộc xe về kho
        solution = np.array(solution, dtype=np.int32)

        if validate.check_depots_sanity(solution):
            if validate.check_capacity_criteria(problem, solution):
                break

    return solution # trả về duy nhất 1 mảng ví dụ : [0,1,2,0,3,4,5,0,6,0]