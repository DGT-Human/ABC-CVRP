import random
import numpy as np
from tqdm import tqdm_notebook
from utils import solution_handler, validate
from algorithm import local_search, random_solution
from algorithm.base import Algorithm
from datetime import datetime


class BeeColony(Algorithm):
    """
    Thuật toán Artificial Bee Colony (ABC) để giải quyết bài toán tối ưu.
    ABC mô phỏng hành vi tìm kiếm thức ăn của đàn ong với ba loại ong:
    - Ong thợ (Employed bees): Tìm kiếm giải pháp xung quanh nguồn thức ăn hiện tại
    - Ong quan sát (Onlooker bees): Chọn nguồn thức ăn dựa trên thông tin từ ong thợ
    - Ong trinh sát (Scout bees): Tìm kiếm nguồn thức ăn mới khi nguồn cũ cạn kiệt
    """

    def __init__(self, problem):
        """
        Khởi tạo thuật toán ABC.
        
        Args:
            problem (dict): Bài toán cần giải quyết, chứa thông tin về:
                - capacity: Sức chứa tối đa
                - distances: Ma trận khoảng cách
                - demands: Nhu cầu của các điểm
        """
        self.problem = problem
        self.history = []          # Lưu lịch sử độ thích hợp
        self.history_alpha = []    # Lưu lịch sử hệ số alpha
        
    @property
    def name(self):
        return 'ArtificialBeeColony'

    def set_params(self, n_epoch=1000, n_initials=20, n_onlookers=10, search_limit=50):
        """
        Thiết lập tham số cho thuật toán.

        Args:
            n_epoch (int): Số vòng lặp tối ưu
            n_initials (int): Số lượng giải pháp ban đầu (số ong thợ)
            n_onlookers (int): Số lượng ong quan sát
            search_limit (int): Số lần tìm kiếm tối đa trước khi bỏ nguồn thức ăn
        """
        self.n_epoch = n_epoch
        self.n_initials = n_initials
        self.n_onlookers = n_onlookers
        self.search_limit = search_limit

    @staticmethod
    def fitness(problem, solution, alpha=0.5, betta=0.5):
        """
        Tính độ thích hợp của một giải pháp.
        
        Args:
            problem (dict): Bài toán cần giải quyết
            solution (list): Giải pháp cần đánh giá
            alpha (float): Hệ số phạt cho việc vượt quá sức chứa
            betta (float): Hệ số phạt cho việc vượt quá độ dài tuyến đường
            
        Returns:
            float: Giá trị độ thích hợp (càng lớn càng tốt)
        """
        cost = validate.compute_solution(problem, solution)
        demands = validate.get_routes_demand(problem, solution)
        capacity_violation = max(demands) - problem['capacity']
        length_violation = 0  # Có thể thêm ràng buộc về độ dài tuyến đường
        
        # Công thức tính độ thích hợp với các hệ số phạt
        return 1 / (cost + alpha * capacity_violation + betta * length_violation)

    from datetime import datetime

    def solve(self, alpha=0.2, betta=0.2, delta=0.01, gen_alpha=1, gen_betta=0.5, callback=None):
        # Khởi tạo quần thể
        population = self._initialize_population(gen_alpha, gen_betta)

        # Đánh giá quần thể ban đầu
        solutions = population['solutions']
        fitnesses = population['fitnesses']
        counters = np.zeros(self.n_initials, dtype=np.int32)

        # Khởi tạo công cụ tìm kiếm
        searchers = {
            'local': local_search.LocalSearch(self.problem),
            'neighbor': local_search.NeighborOperator()
        }

        # Vòng lặp chính của thuật toán
        start_time = datetime.now()
        for itr in range(self.n_epoch):
            # Giai đoạn ong thợ
            for i, solution in enumerate(solutions):
                new_solution = self._employed_bee_search(
                    solution, fitnesses[i], searchers['local'], alpha, betta
                )
                if new_solution['improved']:
                    solutions[i] = new_solution['solution']
                    fitnesses[i] = new_solution['fitness']
                    counters[i] = 0
                else:
                    counters[i] += 1

            # Giai đoạn ong quan sát
            onlooker_results = self._onlooker_bee_search(
                solutions, fitnesses, counters, searchers['neighbor'], alpha, betta
            )
            solutions = onlooker_results['solutions']
            fitnesses = onlooker_results['fitnesses']
            counters = onlooker_results['counters']

            # Giai đoạn ong trinh sát
            solutions = self._scout_bee_search(
                solutions, counters, searchers['neighbor']
            )

            # Điều chỉnh tham số và cập nhật lịch sử
            # alpha = self._adjust_parameters(solutions, alpha, delta)
            self._update_history(fitnesses, alpha)

            # Gọi hàm callback nếu được cung cấp
            if callback:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                remaining_time = (self.n_epoch - itr - 1) * (elapsed_time / (itr + 1))
                callback(itr + 1, self.n_epoch, elapsed_time, remaining_time)

        # Trả về giải pháp tốt nhất
        return self._get_best_solution(solutions, fitnesses)

    def _initialize_population(self, gen_alpha, gen_betta):
        """Khởi tạo quần thể giải pháp ban đầu."""
        solutions = [
            random_solution.generate_solution(
                self.problem,
                alpha=gen_alpha,
                betta=gen_betta,
                patience=100
            ) for _ in range(self.n_initials)
        ]
        
        fitnesses = [
            self.fitness(self.problem, solution, alpha=gen_alpha, betta=gen_betta)
            for solution in solutions
        ]
        
        return {
            'solutions': solutions,
            'fitnesses': fitnesses
        }

    def _employed_bee_search(self, solution, current_fitness, searcher, alpha, betta):
        """Ong thợ tìm kiếm cải thiện cho một giải pháp."""
        searcher.set_params(solution, n_iter=12)
        neighbor, _ = searcher.solve(only_feasible=True)
        neighbor_fitness = self.fitness(self.problem, neighbor, alpha=alpha, betta=betta)
        
        improved = neighbor_fitness > current_fitness
        return {
            'solution': neighbor if improved else solution,
            'fitness': neighbor_fitness if improved else current_fitness,
            'improved': improved
        }

    def _onlooker_bee_search(self, solutions, fitnesses, counters, neighbor_gen, alpha, betta):
        """Ong quan sát tập trung tìm kiếm ở vùng có giải pháp tốt."""
        # Tính xác suất chọn mỗi giải pháp
        total_fitness = sum(fitnesses)
        probabilities = [fit/total_fitness for fit in fitnesses]
        
        # Với mỗi ong quan sát
        for _ in range(self.n_onlookers):
            # Chọn giải pháp dựa trên xác suất
            chosen_idx = np.random.choice(len(solutions), p=probabilities)
            solution = solutions[chosen_idx]
            
            # Tìm giải pháp hàng xóm
            neighbor = neighbor_gen.random_operator(solution, patience=20)
            if not validate.check_capacity_criteria(self.problem, neighbor):
                continue
                
            # Đánh giá giải pháp mới
            neighbor_fitness = self.fitness(self.problem, neighbor, alpha=alpha, betta=betta)
            if neighbor_fitness > fitnesses[chosen_idx]:
                solutions[chosen_idx] = neighbor
                fitnesses[chosen_idx] = neighbor_fitness
                counters[chosen_idx] = 0
            else:
                counters[chosen_idx] += 1
                
        return {
            'solutions': solutions,
            'fitnesses': fitnesses,
            'counters': counters
        }

    def _scout_bee_search(self, solutions, counters, neighbor_gen):
        """Ong trinh sát thay thế các giải pháp không cải thiện bằng lời giải mới tốt hơn."""
        for i, count in enumerate(counters):
            if count >= self.search_limit:
                # Tạo lời giải mới hoàn toàn
                new_solution = random_solution.generate_solution(self.problem, alpha=1.0, betta=0.5, patience=100)
                new_cost = validate.compute_solution(self.problem, new_solution)  # Tính cost của lời giải mới
                old_cost = validate.compute_solution(self.problem, solutions[i])  # Tính cost của lời giải cũ

                # Nếu lời giải mới có chi phí nhỏ hơn, thay thế lời giải cũ
                if new_cost < old_cost:
                    solutions[i] = new_solution
                    counters[i] = 0  # Reset bộ đếm tìm kiếm

        return solutions

    def _adjust_parameters(self, solutions, alpha, delta):
        """Điều chỉnh tham số alpha dựa trên tỷ lệ giải pháp hợp lệ."""
        valid_solutions = sum(1 for s in solutions 
                            if validate.check_capacity_criteria(self.problem, s))
        return alpha - delta if valid_solutions > len(solutions)/2 else alpha + delta

    def _update_history(self, fitnesses, alpha):
        """Cập nhật lịch sử tối ưu."""
        self.history.append(1 / np.mean(fitnesses))
        self.history_alpha.append(alpha)

    def _get_best_solution(self, solutions, fitnesses):
        """Tìm giải pháp tốt nhất từ quần thể cuối cùng."""
        while solutions:
            best_idx = np.argmax(fitnesses)
            if validate.check_solution(self.problem, solutions[best_idx]):
                return solutions[best_idx]
            
            del solutions[best_idx]
            del fitnesses[best_idx]
            
        return None