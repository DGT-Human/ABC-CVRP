import random
import numpy as np
from utils import validate


def generate_solution(problem,
                      patience=50,
                      verbose=False) -> np.ndarray:
    MAXIMUM_PENALTY = 10000000
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
                if  route_demand > problem['capacity']:
                    coef = MAXIMUM_PENALTY
                else:
                    coef = i * len(route) + dist_to_loc
                route_dists.append(coef)

            routes[np.argmin(route_dists)].append(random_loc)
            i_loc.remove(random_loc)

        solution = [loc for route in routes for loc in route]
        solution.append(0)
        solution = np.array(solution, dtype=np.int32)

        if validate.check_depots_sanity(solution):
            if validate.check_capacity_criteria(problem, solution):
                break
    return solution