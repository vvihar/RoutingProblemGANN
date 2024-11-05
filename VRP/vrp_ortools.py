import os
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from pprint import pprint

n_nodes = 101

np.random.seed(0)


def discrete_cmap(N, base_cmap=None):
    base = plt.colormaps.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_vehicle_routes(
    data,
    routes,
    ax,
    greedy,
    markersize=5,
    visualize_demands=False,
    demand_scale=1,
    round_demand=False,
):
    plt.rc("font", family="Helvetica", size=10)

    # print(data)

    depot = data["locations"][0]
    locs = data["locations"][1:]
    demands = np.array(data["demands"])
    capacity = data["capacity"]

    x_dep, y_dep = depot
    ax.plot(x_dep, y_dep, "sk", markersize=markersize * 4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cmap = discrete_cmap(len(routes) + 2, "nipy_spectral")
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number)

        route_demands = [demands[i - 1] for i in r[1:-1]]  # Correctly map indices
        coords = locs[[i - 1 for i in r[1:-1]], :]  # Correctly map indices
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        if not visualize_demands:
            ax.plot(xs, ys, "o", mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0

        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

            height_cap_rect = 0.1
            height_used_rect = 0.1 * total_route_demand / capacity[veh_number]
            height_dem_rect = 0.1 * d / capacity[veh_number]
            y_dem_rect = y + 0.1 * cum_demand / capacity[veh_number]

            cap_rects.append(Rectangle((x, y), 0.01, height_cap_rect))
            used_rects.append(Rectangle((x, y), 0.01, height_used_rect))
            dem_rects.append(Rectangle((x, y_dem_rect), 0.01, height_dem_rect))

            x_prev, y_prev = x, y
            cum_demand += d

        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color=color,
            label="R{}, N({}), C {} / {}, D {:.2f}".format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity[veh_number]) if round_demand else capacity,
                dist,
            ),
        )
        qvs.append(qv)

    ax.legend(handles=qvs, loc=1)
    pc_cap = PatchCollection(
        cap_rects, facecolor="whitesmoke", alpha=1.0, edgecolor="lightgray"
    )
    pc_used = PatchCollection(
        used_rects, facecolor="lightgray", alpha=1.0, edgecolor="lightgray"
    )
    pc_dem = PatchCollection(dem_rects, facecolor="black", alpha=1.0, edgecolor="black")

    if visualize_demands:
        ax.add_collection(pc_cap)
        ax.add_collection(pc_used)
        ax.add_collection(pc_dem)
    plt.show()


def create_data_model():
    """Stores the data for the problem."""
    node_ = np.loadtxt(
        "./test_data/vrp100_test_data.csv", dtype=np.float32, delimiter=","
    )
    demand_ = np.loadtxt(
        "./test_data/vrp100_demand.csv", dtype=np.float32, delimiter=","
    )
    capacity_ = np.loadtxt(
        "./test_data/vrp100_capacity.csv", dtype=np.float32, delimiter=","
    )
    node_, demand_ = node_.reshape(-1, n_nodes, 2), demand_.reshape(-1, n_nodes)

    demand_ = (demand_ * 10).astype(int)
    capacity_ = (capacity_ * 10).astype(int)

    data_size = node_.shape[0]

    x = np.random.randint(1, data_size)

    data = {}
    data["locations"] = node_[x]
    data["demands"] = list(int(d) for d in demand_[x])
    data["num_vehicles"] = 12
    # Create capacity data whose length is equal to the number of vehicles.
    data["capacity"] = list(
        int(c) for c in np.array([capacity_.astype(np.int32)[x]] * data["num_vehicles"])
    )
    data["depot"] = 0
    data["distance_matrix"] = compute_euclidean_distance_matrix(node_[x])
    # pprint(data)
    return data


def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    n_locations = len(locations)
    # dist_matrix = np.zeros((n_locations, n_locations))
    dist_matrix = [[0] * n_locations for _ in range(n_locations)]
    for from_counter, from_node in enumerate(locations):
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                dist_matrix[from_counter][to_counter] = 0
            else:
                dist_matrix[from_counter][to_counter] = int(
                    float(np.linalg.norm(np.array(from_node) - np.array(to_node)))
                    * 1000
                )
    return dist_matrix


def main():
    """Solve the VRP problem."""
    data = create_data_model()

    # pprint(data)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    # print("=====")
    # pprint(distance_matrix)
    # print("=====")
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # print(from_node, to_node, data['distance_matrix'][from_node][to_node])
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return int(data["demands"][from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["capacity"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Get routes.
    routes = []
    total_distance = 0
    total_load = 0
    if solution:
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            plan_output = "Route for vehicle {}:\n".format(vehicle_id)
            route_distance = 0
            route_load = 0
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                route_load += data["demands"][node_index]
                plan_output += " {0} Load({1}) -> ".format(node_index, route_load)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            routes.append(route[1:])
            plan_output += " {0} Load({1})\n".format(
                manager.IndexToNode(index), route_load
            )
            plan_output += "Distance of the route: {}m\n".format(route_distance)
            plan_output += "Load of the route: {}\n".format(route_load)
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print("Total distance of all routes: {}m".format(total_distance))
        print("Total load of all routes: {}".format(total_load))
    else:
        print("No solution found!")

    # Plot routes.
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_vehicle_routes(
        data,
        routes,
        ax,
        greedy=True,
        visualize_demands=False,
        demand_scale=50,
        round_demand=True,
    )


start_time = time.time()
main()
print("Time:", time.time() - start_time)
