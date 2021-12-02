"""Simple travelling salesman problem between cities."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def read_file_and_build_distance(nv,filename):
    with open(filename , mode ="r") as file:
        input_data = [d.strip().split() for d in file.readlines()]
        if len(input_data[0]) != len(input_data[-1]):
            distances = [[None for j in range(nv)] for i in range(nv)]
            for i in range(nv):
                row = 0
                for j in range(i+1,nv):
                    distances[i][j] = int(input_data[i][row])
                    distances[j][i] = distances[i][j]
                    row += 1 
            for i in range(nv):
                distances[i][i] = 0 
        elif len(input_data[0]) == len(input_data[-1]):
            print(2)
            distances = [[None for j in range(nv)] for i in range(nv)]
            for i in range(nv):
                for j in range(nv):
                    distances[i][j] = int(input_data[i][j])
            # print(distances[-1])
    return distances


def create_data_model(nv,filename):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = read_file_and_build_distance(nv,filename)
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    nv = 50
    filename = "distances_entre_villes_{}.txt".format(str(nv))
    data = create_data_model(nv,filename)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    search_parameters.time_limit.seconds = 500
    search_parameters.log_search = True
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution)


if __name__ == '__main__':
    main()