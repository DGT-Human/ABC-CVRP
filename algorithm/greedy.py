import numpy as np
from utils.solution_handler import get_problem, write_solution
from utils.validate import compute_solution, check_solution, check_capacity_criteria

class GreedyCVRP():

    def greedy_cvrp(problem):
        depot = problem['depot_i']
        demands = np.array(problem['demands'])
        capacity = problem['capacity']
        dists = problem['dists']

        num_vehicles = problem['n_trucks']
        vehicles = [[] for _ in range(num_vehicles)]
        vehicle_loads = [0] * num_vehicles
        vehicle_positions = [depot] * num_vehicles

        unvisited = set(range(len(demands)))
        unvisited.remove(depot)

        vehicle_index = 0

        while unvisited:
            best_customer = None
            best_cost = float('inf')

            for customer in unvisited:
                cost = dists[vehicle_positions[vehicle_index]][customer]  # 1️⃣ Tính chi phí di chuyển đến khách hàng này
                demand_ratio = demands[customer] / capacity  # 2️⃣ Xác định tỷ lệ tải trọng so với khả năng chở của xe
                score = cost * (1 - demand_ratio)  # 3️⃣ Kết hợp cả chi phí lẫn tải trọng vào hàm đánh giá

                if vehicle_loads[vehicle_index] + demands[customer] <= capacity and score < best_cost:
                    best_customer = customer
                    best_cost = score

            if best_customer is not None:
                vehicles[vehicle_index].append(best_customer)
                vehicle_loads[vehicle_index] += demands[best_customer]
                vehicle_positions[vehicle_index] = best_customer
                unvisited.remove(best_customer)
            else:
                # Xe đã đầy, quay về depot
                vehicles[vehicle_index].append(depot)
                vehicle_positions[vehicle_index] = depot
                # Nếu còn xe thì chuyển sang xe tiếp theo
                if vehicle_index + 1 < num_vehicles:
                    vehicle_index += 1
                else:
                    print(
                        f"Xe {vehicle_index}: Đã đi qua {vehicles[vehicle_index]} - Tải trọng {vehicle_loads[vehicle_index]}")
                    print(f"Còn lại {len(unvisited)} khách chưa được giao")
                    print(f"Khách hàng chưa được giao: {unvisited}")
                    print(f"Nhu cầu khách hàng chưa giao: {[demands[c] for c in unvisited]}")
                    print(f"⚠ Không thể phục vụ tất cả khách hàng! Còn {len(unvisited)} khách chưa được giao.")
                    break

        # Đảm bảo xe quay về depot
        for route in vehicles:
            if route and route[-1] != depot:
                route.append(depot)

        # ✅ Chuyển tuyến đường thành danh sách tuyến
        solution = [depot] + [node for route in vehicles for node in route]

        # ✅ Kiểm tra tính hợp lệ của lời giải trước khi trả về
        if not check_solution(problem, solution):
            print("🚨 Lời giải không hợp lệ! Thử debug lại...")
            return solution  # Vẫn trả về danh sách tuyến nhưng có cảnh báo

        # ✅ Kiểm tra tổng tải trọng từng xe
        if not check_capacity_criteria(problem, solution):
            print("🚨 Một số tuyến đường vượt quá tải trọng!")
            return solution  # Trả về danh sách tuyến nhưng có cảnh báo

        return solution  # ✅ Giữ cách return như cũ

