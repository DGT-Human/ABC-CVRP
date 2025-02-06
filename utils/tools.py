import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations
from tqdm import tqdm, tqdm_notebook

def get_problem(path: str) -> dict:
    with open(path, 'r') as f:
        file = f.read().splitlines()

    problem_dict = {}
    section = None
    for i, row in enumerate(file):
        if 'COMMENT' in row:
            _, comment = row.split(':', 1)
            problem_dict['n_trucks'] = int(comment.split('No of trucks:')[1].split(',')[0])
            problem_dict['optimal'] = int(comment.split('Optimal value:')[1].split(')')[0])
        elif 'CAPACITY' in row:
            problem_dict['capacity'] = int(row.split(':')[1].strip())
        elif 'NODE_COORD_SECTION' in row:
            section = 'NODE_COORD_SECTION'
            problem_dict['locations'] = []
        elif 'DEMAND_SECTION' in row:
            section = 'DEMAND_SECTION'
            problem_dict['demands'] = []
        elif section == 'NODE_COORD_SECTION':
            node_id, x, y = row.split()
            problem_dict['locations'].append((float(x), float(y)))
        elif section == 'DEMAND_SECTION':
            parts = row.split()
            if len(parts) >= 2:
                node_id, demand = parts[:2]
                if int(demand) == 0:
                    problem_dict['depot_i'] = len(problem_dict['demands'])
                problem_dict['demands'].append(float(demand))

    problem_dict['n_locations'] = len(problem_dict['locations'])
    assert problem_dict['n_locations'] == len(problem_dict['demands'])

    locations = problem_dict['locations']
    n = problem_dict['n_locations']
    dists = np.zeros((n, n), dtype=np.float32)
    for (i, loc1), (j, loc2) in combinations(enumerate(locations), 2):
        if i != j:
            dists[i, j] = dists[j, i] = np.linalg.norm(np.array(loc1) - np.array(loc2))
    problem_dict['dists'] = dists

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