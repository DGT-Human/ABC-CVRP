import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import utils.solution_handler as tools
import utils.validate as common
import algorithm.bee_colony as bee_colony
import utils.visualize as visualize
from tqdm import tqdm
import glob
import os
from algorithm import random_solution

class VRPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Routing Problem Solver")

        # Variables
        self.problem = None
        self.initial_solution = None
        self.abc_solution = None

        # Create notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create main tab
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Main")

        # Create main frame
        self.main_frame = ttk.Frame(main_tab, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create input frame
        self.create_input_frame()

        # Create parameters frame
        self.create_parameters_frame()

        # Create output frame
        self.create_output_frame()

        # Create visualization frame
        self.create_viz_frame()

        # Create benchmark tab
        self.create_benchmark_tab()
        self.create_load_info_tab()


    def create_input_frame(self):
        # Input Frame
        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Load benchmark button
        ttk.Button(input_frame, text="Load Benchmark", command=self.load_benchmark).grid(row=0, column=0, padx=5,
                                                                                         pady=5)

        # Generate initial solution button
        ttk.Button(input_frame, text="Generate Initial Solution", command=self.generate_initial_solution).grid(row=1,
                                                                                                               column=0,
                                                                                                               padx=5,
                                                                                                               pady=5)

        ttk.Button(input_frame, text="Show File Content", command=self.show_file_content).grid(row=2, column=0, padx=5,
                                                                                               pady=5)

        # Display selected file
        self.file_label = ttk.Label(input_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, padx=5, pady=5)

    def create_parameters_frame(self):
        # Parameters Frame
        param_frame = ttk.LabelFrame(self.main_frame, text="ABC Parameters", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Parameters entries
        ttk.Label(param_frame, text="Number of Epochs:").grid(row=0, column=0, padx=5, pady=2)
        self.epochs_var = tk.StringVar(value="400")
        ttk.Entry(param_frame, textvariable=self.epochs_var).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="Number of Initial Solutions:").grid(row=1, column=0, padx=5, pady=2)
        self.initials_var = tk.StringVar(value="40")
        ttk.Entry(param_frame, textvariable=self.initials_var).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="Number of Onlookers:").grid(row=2, column=0, padx=5, pady=2)
        self.onlookers_var = tk.StringVar(value="20")
        ttk.Entry(param_frame, textvariable=self.onlookers_var).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="Search Limit:").grid(row=3, column=0, padx=5, pady=2)
        self.search_limit_var = tk.StringVar(value="10")
        ttk.Entry(param_frame, textvariable=self.search_limit_var).grid(row=3, column=1, padx=5, pady=2)

        # Solve button
        ttk.Button(param_frame, text="Solve VRP", command=self.solve_vrp).grid(row=4, column=0, pady=10)
        ttk.Button(param_frame, text="Test Optimization", command=self.prompt_test_parameters).grid(row=4, column=1,
                                                                                                    pady=10)
        ttk.Button(param_frame, text="Check Stability", command=self.prompt_stability_test).grid(row=5, column=0, columnspan=2, pady=10)


    def create_output_frame(self):
        # Output Frame
        output_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        output_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Results labels
        self.initial_cost_label = ttk.Label(output_frame, text="Initial Cost: -")
        self.initial_cost_label.grid(row=0, column=0, padx=5, pady=2)

        self.initial_feasible_label = ttk.Label(output_frame, text="Initial Feasible: -")
        self.initial_feasible_label.grid(row=1, column=0, padx=5, pady=2)

        self.abc_cost_label = ttk.Label(output_frame, text="ABC Cost: -")
        self.abc_cost_label.grid(row=2, column=0, padx=5, pady=2)

        self.abc_time_label = ttk.Label(output_frame, text="ABC Time: -")
        self.abc_time_label.grid(row=3, column=0, padx=5, pady=2)

        self.abc_feasible_label = ttk.Label(output_frame, text="ABC Feasible: -")
        self.abc_feasible_label.grid(row=4, column=0, padx=5, pady=2)

        ttk.Button(output_frame, text="Load Solution", command=self.load_solution).grid(row=5, column=0, padx=5, pady=2)

    def create_viz_frame(self):
        # Routes Frame
        self.routes_frame = ttk.LabelFrame(self.main_frame, text="Solution Routes", padding="5")
        self.routes_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Text widget để hiển thị routes
        self.routes_text = tk.Text(self.routes_frame, wrap=tk.WORD, font=('Courier', 11), width=40,
                                   height=8)  # Tăng width từ 35 lên 40
        self.routes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar cho text widget
        scrollbar = ttk.Scrollbar(self.routes_frame, command=self.routes_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.routes_text.config(yscrollcommand=scrollbar.set)

        # Visualization Frame
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="5")
        self.viz_frame.grid(row=1, column=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Button frame cho nút Clear
        button_frame = ttk.Frame(self.viz_frame)
        button_frame.grid(row=0, column=0, sticky='w', padx=5, pady=2)

        # Clear button ở bên trái
        ttk.Button(button_frame, text="Clear hình", command=self.clear_visualization).pack(side=tk.LEFT)

        # Create figure với kích thước rộng hơn
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Tăng chiều ngang từ 10 lên 12
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Điều chỉnh khoảng cách giữa các subplot
        plt.subplots_adjust(wspace=0.3)

        # Configure grid weights cho viz_frame
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(1, weight=1)


    def load_benchmark(self):
        filename = filedialog.askopenfilename(
            title="Select Benchmark File",
            filetypes=[("VRP files", "*.vrp"), ("All files", "*.*")]
        )
        if filename:
            self.problem = tools.get_problem(filename)
            self.file_label.config(text=filename.split("/")[-1])
            self.update_visualization()

    def generate_initial_solution(self):
        if self.problem is None:
            messagebox.showerror("Error", "Please load a benchmark first!")
            return

        self.initial_solution = random_solution.generate_solution(self.problem, verbose=False)
        initial_cost = common.compute_solution(self.problem, self.initial_solution)
        is_feasible = common.check_solution(self.problem, self.initial_solution)

        self.initial_cost_label.config(text=f"Initial Cost: {initial_cost:.2f}")
        self.initial_feasible_label.config(text=f"Initial Feasible: {is_feasible}")

        self.update_visualization()

    def solve_vrp(self):
        if self.problem is None:
            messagebox.showerror("Error", "Please load a benchmark first!")
            return

        try:
            # Get parameters
            n_epoch = int(self.epochs_var.get())
            n_initials = int(self.initials_var.get())
            n_onlookers = int(self.onlookers_var.get())
            search_limit = int(self.search_limit_var.get())

            # Initialize ABC
            ABC = bee_colony.BeeColony(self.problem)
            ABC.set_params(
                n_epoch=n_epoch,
                n_initials=n_initials,
                n_onlookers=n_onlookers,
                search_limit=search_limit
            )

            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Optimization Progress")
            progress_label = ttk.Label(progress_window, text="Optimizing...")
            progress_label.pack()
            progress_bar = ttk.Progressbar(progress_window, length=400, mode='determinate')
            progress_bar.pack()

            def update_progress(current, total, elapsed, remaining):
                progress = int((current / total) * 100)
                progress_bar['value'] = progress
                progress_label.config(
                    text=f"Optimizing: {progress}%|{'█' * (progress // 5)}{' ' * (20 - progress // 5)}| {current}/{total} [{elapsed:.0f}s<{remaining:.0f}s, {(total / elapsed):.2f}it/s]")
                progress_window.update()

            # Solve
            start_time = datetime.now()
            self.abc_solution = ABC.solve(callback=update_progress)
            solve_time = (datetime.now() - start_time).total_seconds()

            # Close progress window
            progress_window.destroy()

            # Calculate results
            abc_cost = common.compute_solution(self.problem, self.abc_solution)
            is_feasible = common.check_solution(self.problem, self.abc_solution)

            # Update results
            self.abc_cost_label.config(text=f"ABC Cost: {abc_cost:.2f}")
            self.abc_time_label.config(text=f"ABC Time: {solve_time:.2f}s")
            self.abc_feasible_label.config(text=f"ABC Feasible: {is_feasible}")

            # Update visualization
            self.update_visualization()
            self.update_routes()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_visualization(self):
        self.ax1.clear()
        self.ax2.clear()

        if self.problem is not None:
            if self.initial_solution is not None:
                visualize.visualize_problem(self.problem, self.initial_solution, ax=self.ax1)
                self.ax1.set_title("Initial Solution")

            if self.abc_solution is not None:
                visualize.visualize_problem(self.problem, self.abc_solution, ax=self.ax2)
                self.ax2.set_title("ABC Solution")

            self.canvas.draw()


    def update_routes(self):
        if self.abc_solution is not None:
            routes = common.get_routes(self.abc_solution)
            self.routes_text.delete(1.0, tk.END)

            for i, route in enumerate(routes, start=1):
                route_str = f"Route #{i}: " + " → ".join(str(node) for node in route) + "\n"
                self.routes_text.insert(tk.END, route_str)
        else:
            self.routes_text.delete(1.0, tk.END)
            self.routes_text.insert(tk.END, "No solution available.")

    def create_benchmark_tab(self):
        benchmark_frame = ttk.Frame(self.notebook)
        self.notebook.add(benchmark_frame, text="Benchmark")

        # Create input frame to group controls together
        input_frame = ttk.Frame(benchmark_frame)
        input_frame.grid(row=0, column=0, columnspan=3, sticky='w', padx=3, pady=3)

        # Group the input controls closer together
        ttk.Label(input_frame, text="Benchmark Folder:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self.benchmark_folder_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.benchmark_folder_var, width=50).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(input_frame, text="Browse", command=self.browse_benchmark_folder).grid(row=0, column=2, padx=2,
                                                                                          pady=2)

        ttk.Label(input_frame, text="Number of Epochs:").grid(row=1, column=0, padx=2, pady=2, sticky=tk.W)
        self.benchmark_epochs_var = tk.StringVar(value="300")
        ttk.Entry(input_frame, textvariable=self.benchmark_epochs_var, width=10).grid(row=1, column=1, padx=2, pady=2,
                                                                                      sticky=tk.W)

        ttk.Label(input_frame, text="Number of Onlookers:").grid(row=2, column=0, padx=2, pady=2, sticky=tk.W)
        self.benchmark_onlookers_var = tk.StringVar(value="5")
        ttk.Entry(input_frame, textvariable=self.benchmark_onlookers_var, width=10).grid(row=2, column=1, padx=2, pady=2,
                                                                               sticky=tk.W)

        # Run button with reduced padding
        ttk.Button(input_frame, text="Run Benchmark", command=self.run_benchmark).grid(row=3, column=0, columnspan=3,
                                                                                       padx=2, pady=5)

        # Create table frame
        table_frame = ttk.Frame(benchmark_frame)
        table_frame.grid(row=1, column=0, columnspan=3, sticky='nsew', padx=3, pady=3)

        # Benchmark table
        self.benchmark_table = ttk.Treeview(table_frame, columns=(
            "Benchmark", "Locations", "Trucks", "Capacity", "Optimal Cost",
            "ABC Cost", "ABC Time", "Error", "Feasible"),
                                            show="headings")

        # Configure headings
        columns = ["Benchmark", "Locations", "Trucks", "Capacity", "Optimal Cost",
                   "ABC Cost", "ABC Time", "Error", "Feasible"]
        for col in columns:
            self.benchmark_table.heading(col, text=col)

        self.benchmark_table.grid(row=0, column=0, sticky='nsew')

        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.benchmark_table.yview)
        self.benchmark_table.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky='ns')

        # Configure grid weights for proper expansion
        benchmark_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

    def browse_benchmark_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.benchmark_folder_var.set(folder_selected)

    def run_benchmark(self):
        benchmark_folder = self.benchmark_folder_var.get()
        n_epochs = int(self.benchmark_epochs_var.get())
        n_onlookers = int(self.benchmark_onlookers_var.get())

        benchmark_files = glob.glob(f"{benchmark_folder}/*.vrp")

        # Xóa nội dung cũ trong khung văn bản
        self.load_info_text.delete(1.0, tk.END)

        # Tạo cửa sổ loading
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Loading...")

        # Tạo tiến trình thanh
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(loading_window, variable=progress_var, maximum=len(benchmark_files))
        progress_bar.pack(fill=tk.X, padx=10, pady=10)

        # Tạo nhãn thông báo
        loading_label = ttk.Label(loading_window, text="Running benchmark...")
        loading_label.pack(padx=10, pady=10)

        loading_window.update()

        info_dict = {
            "benchmark": [],
            "n_locations": [],
            "n_trucks": [],
            "capacity": [],
            "optimal_cost": [],
            "ABC_cost": [],
            "ABC_time": [],
            "is_feasible": [],
            "error": [],
            "abc_solution": [],
            "abc_epochs": [],
            "abc_employers": [],
            "abc_onlookers": [],
            "abc_search_limit": []
        }

        for i, benchmark_file in enumerate(benchmark_files):
            problem = tools.get_problem(benchmark_file)
            bench_name = os.path.basename(benchmark_file)

            self.load_info_text.insert(tk.END, f"#{i} {bench_name} ...\n")
            self.load_info_text.update_idletasks()

            ABC = bee_colony.BeeColony(problem)
            ABC.set_params(
                n_epoch=n_epochs,
                n_initials=problem["n_locations"],
                n_onlookers=n_onlookers,
                search_limit=problem["n_locations"]
            )

            start_time = datetime.now()
            abc_solution = ABC.solve()
            end_time = (datetime.now() - start_time).total_seconds()

            abc_cost = common.compute_solution(problem, abc_solution)
            is_feasible = common.check_solution(problem, abc_solution)
            error = (abc_cost - problem["optimal"]) / problem["optimal"]

            self.load_info_text.insert(tk.END,
                                       f"epoch: {ABC.n_epoch} initials: {ABC.n_initials} search_limit: {ABC.search_limit}\n")
            self.load_info_text.update_idletasks()

            info_dict["benchmark"].append(bench_name)
            info_dict["n_locations"].append(problem["n_locations"])
            info_dict["n_trucks"].append(problem["n_trucks"])
            info_dict["capacity"].append(problem["capacity"])
            info_dict["optimal_cost"].append(problem["optimal"])
            info_dict["ABC_cost"].append(abc_cost)
            info_dict["ABC_time"].append(end_time)
            info_dict["is_feasible"].append(is_feasible)
            info_dict["error"].append(error)
            info_dict["abc_solution"].append(abc_solution)
            info_dict["abc_epochs"].append(ABC.n_epoch)
            info_dict["abc_employers"].append(ABC.n_initials)
            info_dict["abc_onlookers"].append(ABC.n_onlookers)
            info_dict["abc_search_limit"].append(ABC.search_limit)

            # Cập nhật tiến trình thanh
            progress_var.set(i + 1)
            loading_window.update_idletasks()

        # Đóng cửa sổ loading
        loading_window.destroy()

        # Xóa các hàng cũ trong bảng
        for row in self.benchmark_table.get_children():
            self.benchmark_table.delete(row)

        # Thêm các hàng mới vào bảng
        for i in range(len(info_dict["benchmark"])):
            self.benchmark_table.insert("", tk.END, values=(
                info_dict["benchmark"][i],
                info_dict["n_locations"][i],
                info_dict["n_trucks"][i],
                info_dict["capacity"][i],
                info_dict["optimal_cost"][i],
                info_dict["ABC_cost"][i],
                info_dict["ABC_time"][i],
                info_dict["error"][i],
                info_dict["is_feasible"][i]
            ))

    def create_load_info_tab(self):
        load_info_tab = ttk.Frame(self.notebook)
        self.notebook.add(load_info_tab, text="Load Information")

        # Tạo khung văn bản để hiển thị thông tin load
        load_info_frame = ttk.LabelFrame(load_info_tab, text="Load Information", padding="10")
        load_info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.load_info_text = tk.Text(load_info_frame, wrap=tk.WORD, width=80, height=20)
        self.load_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(load_info_frame, orient=tk.VERTICAL, command=self.load_info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.load_info_text.config(yscrollcommand=scrollbar.set)

    def clear_visualization(self):
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

    def show_file_content(self):
        if self.problem is None:
            messagebox.showerror("Error", "Please load a benchmark file first!")
            return

        # Create new window for file content
        content_window = tk.Toplevel(self.root)
        content_window.title("File Content")
        content_window.geometry("600x400")

        # Create text widget
        text_widget = tk.Text(content_window, wrap=tk.WORD, font=('Courier', 11))
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(content_window, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

        # Display problem information
        text_widget.insert(tk.END, f"Problem Information:\n")
        text_widget.insert(tk.END, f"{'-' * 50}\n")
        text_widget.insert(tk.END, f"Number of locations: {self.problem['n_locations']}\n")
        text_widget.insert(tk.END, f"Number of trucks: {self.problem['n_trucks']}\n")
        text_widget.insert(tk.END, f"Truck capacity: {self.problem['capacity']}\n")
        text_widget.insert(tk.END, f"Optimal cost: {self.problem['optimal']}\n")
        text_widget.insert(tk.END, f"\nLocations:\n")
        text_widget.insert(tk.END, f"{'-' * 50}\n")

        # Display locations
        for i, location in enumerate(self.problem['locations']):
            if i == self.problem['depot_i']:
                text_widget.insert(tk.END, f"Location {i} (DEPOT): {location}\n")
            else:
                text_widget.insert(tk.END, f"Location {i}: {location}\n")

        # Display demands
        text_widget.insert(tk.END, f"\nDemands:\n")
        text_widget.insert(tk.END, f"{'-' * 50}\n")
        for i, demand in enumerate(self.problem['demands']):
            text_widget.insert(tk.END, f"Location {i}: {demand}\n")

        # Make text widget read-only
        text_widget.config(state='disabled')

    def load_solution(self):
        if self.problem is None:
            messagebox.showerror("Error", "Please load a benchmark first!")
            return

        sol_filename = filedialog.askopenfilename(
            title="Select Solution File",
            filetypes=[("Solution files", "*.sol"), ("All files", "*.*")]
        )
        if sol_filename:
            with open(sol_filename, 'r') as file:
                sol_content = file.read()

            # Parse the solution content and extract the routes
            routes = []
            cost = None
            for line in sol_content.split('\n'):
                if line.startswith('Route'):
                    route = [int(node) for node in line.split(':')[1].split()]
                    routes.append(route)
                elif line.startswith('Cost'):
                    cost = float(line.split()[-1])

            # Convert the routes to a solution format compatible with the problem
            sol_solution = [0] * self.problem['n_locations']
            for i, route in enumerate(routes, start=1):
                for node in route:
                    sol_solution[node] = i

            # Create a new window for visualization
            sol_window = tk.Toplevel(self.root)
            sol_window.title("Solution Visualization")

            # Create frame cho visualization và routes
            viz_frame = ttk.Frame(sol_window, padding="5")
            viz_frame.pack(fill=tk.BOTH, expand=True)

            # Create a figure and axes for the solution visualization
            sol_fig, sol_ax = plt.subplots(figsize=(10, 8))

            # Visualize solution with lines connecting nodes in routes
            locations = np.array(self.problem['locations'])
            depot_idx = self.problem['depot_i']
            depot_loc = locations[depot_idx]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Plot depot
            sol_ax.scatter(depot_loc[0], depot_loc[1], c='red', s=100, marker='s', label='Depot')

            # Plot routes with lines connecting nodes
            for i, route in enumerate(routes):
                route_coords = locations[route]
                # Add lines connecting nodes in route
                route_coords_with_depot = np.vstack(([depot_loc], route_coords, [depot_loc]))
                sol_ax.plot(route_coords_with_depot[:, 0], route_coords_with_depot[:, 1],
                            c=colors[i % len(colors)], linewidth=2, label=f'Route {i + 1}')
                # Add nodes
                sol_ax.scatter(route_coords[:, 0], route_coords[:, 1],
                               c=colors[i % len(colors)], s=50)
                # Add node labels
                for coord, node in zip(route_coords, route):
                    sol_ax.annotate(str(node), (coord[0], coord[1]),
                                    xytext=(5, 5), textcoords='offset points')

            sol_ax.set_title(f"Solution Visualization (Cost: {cost})")
            sol_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Create a canvas for the solution visualization
            sol_canvas = FigureCanvasTkAgg(sol_fig, master=viz_frame)
            sol_canvas.draw()
            sol_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Display routes
            routes_frame = ttk.LabelFrame(sol_window, text="Routes", padding="5")
            routes_frame.pack(fill=tk.X, padx=5, pady=5)

            routes_text = tk.Text(routes_frame, wrap=tk.WORD, font=('Courier', 11), height=10)
            routes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Add scrollbar for routes
            scrollbar = ttk.Scrollbar(routes_frame, orient=tk.VERTICAL, command=routes_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            routes_text.config(yscrollcommand=scrollbar.set)

            for i, route in enumerate(routes, start=1):
                route_str = f"Route #{i}: " + " → ".join(str(node) for node in route) + "\n"
                routes_text.insert(tk.END, route_str)

            if cost:
                routes_text.insert(tk.END, f"\nTotal Cost: {cost}")

            routes_text.config(state='disabled')

    def prompt_test_parameters(self):
        self.test_window = tk.Toplevel(self.root)
        self.test_window.title("Test Parameters")

        ttk.Label(self.test_window, text="Start Epochs:").grid(row=0, column=0, padx=5, pady=5)
        self.start_epochs_var = tk.StringVar(value="50")
        ttk.Entry(self.test_window, textvariable=self.start_epochs_var).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.test_window, text="End Epochs:").grid(row=1, column=0, padx=5, pady=5)
        self.end_epochs_var = tk.StringVar(value="1000")
        ttk.Entry(self.test_window, textvariable=self.end_epochs_var).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.test_window, text="Step:").grid(row=2, column=0, padx=5, pady=5)
        self.step_epochs_var = tk.StringVar(value="50")
        ttk.Entry(self.test_window, textvariable=self.step_epochs_var).grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(self.test_window, text="Run Test", command=self.run_test_optimization).grid(row=3, column=0,
                                                                                               columnspan=2, pady=10)

    def run_test_optimization(self):
        self.test_window.destroy()
        if self.problem is None:
            messagebox.showerror("Error", "Please load a benchmark first!")
            return

        start_epochs = int(self.start_epochs_var.get())
        end_epochs = int(self.end_epochs_var.get())
        step_epochs = int(self.step_epochs_var.get())

        results = []
        for n_epoch in range(start_epochs, end_epochs + 1, step_epochs):
            self.epochs_var.set(str(n_epoch))
            self.solve_vrp()

            abc_cost = float(self.abc_cost_label.cget("text").split(":")[-1])
            abc_time = float(self.abc_time_label.cget("text").split(":")[-1].replace("s", ""))

            results.append((n_epoch, abc_cost, abc_time))

        self.show_chart_window(results)

    def show_chart_window(self, results):
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Optimization Results")

        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        epochs, costs, times = zip(*results)

        axs[0].plot(epochs, costs, marker='o', linestyle='-', label='ABC Cost')
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("ABC Cost")
        axs[0].set_title("Cost Optimization Over Epochs")
        axs[0].legend()

        axs[1].plot(epochs, times, marker='s', linestyle='-', color='red', label='ABC Time (s)')
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Time (s)")
        axs[1].set_title("Execution Time Over Epochs")
        axs[1].legend()

        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def prompt_stability_test(self):
        self.stability_window = tk.Toplevel(self.root)
        self.stability_window.title("Stability Test Parameters")

        ttk.Label(self.stability_window, text="Number of Runs:").grid(row=0, column=0, padx=5, pady=5)
        self.num_runs_var = tk.StringVar(value="10")
        ttk.Entry(self.stability_window, textvariable=self.num_runs_var).grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(self.stability_window, text="Run Test", command=self.run_stability_test).grid(row=1, column=0,
                                                                                                 columnspan=2, pady=10)

    def run_stability_test(self):
        self.stability_window.destroy()
        if self.problem is None:
            messagebox.showerror("Error", "Please load a benchmark first!")
            return

        num_runs = int(self.num_runs_var.get())
        results = []

        for _ in range(num_runs):
            self.solve_vrp()
            abc_cost = float(self.abc_cost_label.cget("text").split(":")[-1])
            results.append(abc_cost)

        self.show_stability_line_chart(results)

    def show_stability_line_chart(self, results):
        stability_window = tk.Toplevel(self.root)
        stability_window.title("Stability Test Results")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(results) + 1), results, marker='o', linestyle='-', color='b', label='ABC Cost')
        mean_value = np.mean(results)
        ax.axhline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

        ax.set_title("ABC Cost Stability Analysis")
        ax.set_xlabel("Run Number")
        ax.set_ylabel("ABC Cost")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=stability_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = VRPGUI(root)
    root.mainloop()