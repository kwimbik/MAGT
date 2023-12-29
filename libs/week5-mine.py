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


def regret_self_play(matrix_player_1,matrix_player_2, num_iterations):
    num_strategies = matrix_player_1.shape[0]
    
    row_regret = np.ones(num_strategies)
    column_regret = np.ones(num_strategies)

    for i in range(num_iterations):
        # Normalize regrets
        row_regret_normalized  = [float(i)/sum(row_regret) for i in row_regret]
        column_regret_normalized = [float(i)/sum(column_regret) for i in column_regret]
        print("row_regret,normalized:", row_regret_normalized)
        print("column_regret_normalized:", column_regret_normalized)
        
        # Compute row deviating vectors
        row_deviating_vector = np.dot(matrix_player_1,column_regret_normalized)
        column_deviating_vector = np.dot(matrix_player_2,row_regret_normalized)
        
        # Compute expected utility
        row_expected_utility = np.dot(row_deviating_vector, row_regret_normalized)
        column_expected_utility = np.dot(column_deviating_vector,column_regret_normalized)

        # Calculate regret gains
        row_regret_gain = np.zeros(num_strategies)
        column_regret_gain = np.zeros(num_strategies)
        for i in range(len(row_deviating_vector)):
            strategy = row_deviating_vector[i]
            dif = strategy - row_expected_utility
            row_regret_gain[i] = max(0, dif)
        print("Row expected utility:", row_expected_utility)
        print("row_regret_gain:", row_regret_gain)
            
        for i in range(len(column_deviating_vector)):
            strategy = column_deviating_vector[i]
            dif = strategy - row_expected_utility
            column_regret_gain[i] = max(0, dif)

        # Update regrets
        row_regret = row_regret + row_regret_gain
        column_regret = column_regret + column_regret_gain

    return row_regret_normalized, column_regret_normalized

# Run regret matching self-play
matrix1 = np.array([[15, 2, -1], [-1, -4, 1], [1, -2, 2]])
matrix2 = np.array([[1, 23, -1], [11, -4, 1], [5, -2, 2]])
row_regret, column_regret = regret_self_play(matrix1,matrix2, 2)
print(row_regret)
print(column_regret)
