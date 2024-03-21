import numpy as np
import matplotlib.pyplot as plt
# import libs.week1 as week1


def compute_deltas(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> np.array:
    """Computer how much the players could improve if they were to switch to a best response"""    
    current_utility_row = np.dot(row_strategy, np.dot(matrix, column_strategy))
    current_utility_column = np.dot(column_strategy.T, np.dot(matrix.T, row_strategy.T))

    # Calculate the best response utility for the row player
    best_response_utilities_row = np.dot(matrix, column_strategy)
    best_response_utility_row = np.max(best_response_utilities_row)

    # Calculate the best response utility for the column player
    best_response_utilities_column = np.dot(matrix.T, row_strategy.T)
    best_response_utility_column = np.min(best_response_utilities_column) #??????

    # Calculate the deltas
    delta_row = best_response_utility_row - current_utility_row
    delta_column = best_response_utility_column - current_utility_column

    return np.array([delta_row, delta_column]).flatten()



def compute_exploitability(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
        # Compute the deltas for both players
    deltas = compute_deltas(matrix, row_strategy, column_strategy)

    # Compute NASHCONV as the sum of the deltas
    nash_conv = np.sum(deltas)

    # Compute exploitability by dividing NASHCONV by the number of players (2)
    exploitability = nash_conv / 2

    return exploitability


def compute_epsilon(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Computes epsilon as defined for epsilon-Nash equilibrium"""
    deltas = compute_deltas(matrix, row_strategy, column_strategy)

    # The epsilon value is the maximum of these deltas
    epsilon = np.max(deltas)

    return epsilon


def compute_exploitability_zero_sum(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Compute exploitability for a zero-sum game"""
    # Calculate the best response utility for the row player
    best_response_utilities_row = np.dot(matrix, column_strategy)
    best_response_utility_row = np.max(best_response_utilities_row)

    # Calculate the current utility for the row player
    current_utility_row = np.dot(row_strategy, np.dot(matrix, column_strategy))

    # The exploitability is the improvement the row player can make
    exploitability = best_response_utility_row - current_utility_row

    return exploitability



def best_response(matrix, opponent_strategy):
    # Calculate the best response to an opponent's strategy
    utilities = matrix @ opponent_strategy
    return np.argmax(utilities)

def update_average_strategy(average_strategy, new_strategy, iteration):
    # Update the average strategy
    return (average_strategy * iteration + new_strategy) / (iteration + 1)


def naive_self_play(matrix, num_iterations):
    num_strategies = matrix.shape[0]
    row_strategy = np.zeros(num_strategies)
    column_strategy = np.zeros(num_strategies)

    row_average_strategy = np.zeros(num_strategies)
    column_average_strategy = np.zeros(num_strategies)

    exploitabilities = []

    for i in range(num_iterations):
        # Row player best responds to the last strategy of the column player
        row_response = best_response(matrix, column_strategy)
        row_strategy[:] = 0
        row_strategy[row_response] = 1

        # Column player best responds to the last strategy of the row player
        column_response = best_response(-matrix.T, row_strategy)
        column_strategy[:] = 0
        column_strategy[column_response] = 1

        # Update average strategies
        row_average_strategy = update_average_strategy(row_average_strategy, row_strategy, i)
        column_average_strategy = update_average_strategy(column_average_strategy, column_strategy, i)

        # Compute exploitability of the average strategy
        exploitability = compute_exploitability_zero_sum(matrix, row_average_strategy, column_average_strategy)
        exploitabilities.append(exploitability)

    return row_average_strategy, column_average_strategy, exploitabilities

# Run fictitious and naive self-play
#matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
#_, _, fictitious_exploitabilities = naive_self_play(matrix, 100)
#print(fictitious_exploitabilities)
