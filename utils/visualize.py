import numpy as np
import matplotlib.pyplot as plt

def plot_fitness_history(history, alpha=None, figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history, label='Fitness')
    if alpha is not None:
        ax.plot(alpha, label='Alpha')
    ax.set_ylabel('Fitness Value')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')  # Chuyển lên góc phải trên
    ax.grid(True)
    return fig, ax


def plot_problem_locations(locations, depot_index, figsize=(10, 6), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x, y = zip(*locations)
    ax.plot(x, y, 'o', label='Locations', color='black', markersize=12)
    depot_x, depot_y = locations[depot_index]
    ax.plot(depot_x, depot_y, 'o', label='Depot', markersize=13, color='blue')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')  # Chuyển lên góc phải trên
    ax.grid(True)
    return fig, ax

def plot_solution_routes(locations, solution, depot_indices, figsize=(10, 6), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax = plot_problem_locations(locations, depot_indices[0], figsize, ax=ax)[1]

    for i, (start, end) in enumerate(zip(depot_indices[:-1], depot_indices[1:])):
        route_indices = solution[start:end + 1]
        route_locations = [locations[i] for i in route_indices]
        x, y = zip(*route_locations)
        ax.plot(x, y, '--', label=f'Route {i + 1}')

    # Tạo khung chú thích với nền trắng trong suốt
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True, framealpha=0.7, facecolor='white')

    return fig, ax, legend

def visualize_problem(problem, solution=None, annotate=True, figsize=(10, 6), ax=None):
    locations = problem['locations']
    depot_index = problem['depot_i']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    legend = None
    if solution is not None:
        depot_indices = [i for i, node in enumerate(solution) if node == 0]
        fig, ax, legend = plot_solution_routes(locations, solution, depot_indices, ax=ax)
    else:
        fig, ax = plot_problem_locations(locations, depot_index, ax=ax)

    if annotate:
        for i, (x, y) in enumerate(locations):
            ax.annotate(str(i), (x, y), color='white', ha='center', va='center')

    return fig, ax, legend
