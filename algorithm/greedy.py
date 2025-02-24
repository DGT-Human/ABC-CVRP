import numpy as np
from utils.solution_handler import get_problem, write_solution
from utils.validate import compute_solution, check_solution

class GreedyCVRP():

    def greedy_cvrp(problem):
        depot = problem['depot_i']
        demands = np.array(problem['demands'])
        capacity = problem['capacity']
        dists = problem['dists']

        # Khởi tạo danh sách xe tải
        num_vehicles = problem['n_trucks']
        vehicles = [[] for _ in range(num_vehicles)]
        vehicle_loads = [0] * num_vehicles
        vehicle_positions = [depot] * num_vehicles

        # Danh sách khách hàng chưa được phục vụ
        unvisited = set(range(len(demands)))
        unvisited.remove(depot)

        vehicle_index = 0

        while unvisited:
            best_customer = None
            best_cost = float('inf')

            # Tìm khách hàng gần nhất có thể phục vụ
            for customer in unvisited:
                cost = dists[vehicle_positions[vehicle_index]][customer]
                if vehicle_loads[vehicle_index] + demands[customer] <= capacity and cost < best_cost:
                    best_customer = customer
                    best_cost = cost

            if best_customer is not None:
                # Thêm khách hàng vào tuyến đường hiện tại
                vehicles[vehicle_index].append(best_customer)
                vehicle_loads[vehicle_index] += demands[best_customer]
                vehicle_positions[vehicle_index] = best_customer
                unvisited.remove(best_customer)
            else:
                # Không thể thêm khách hàng nào nữa, quay lại depot và chuyển sang xe tiếp theo
                vehicles[vehicle_index].append(depot)
                vehicle_positions[vehicle_index] = depot

                if vehicle_index + 1 < num_vehicles:
                    vehicle_index += 1
                else:
                    print("Không thể phục vụ tất cả khách hàng với số xe hiện tại.")
                    break

        # Đảm bảo tất cả xe quay về depot
        for route in vehicles:
            if route and route[-1] != depot:
                route.append(depot)

        # Chuyển kết quả thành danh sách tuyến đường
        solution = [depot] + [node for route in vehicles for node in route]
        return solution

