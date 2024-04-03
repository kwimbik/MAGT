import numpy as np
import matplotlib.pyplot as plt

# import libs.week1 as week1


def compute_deltas(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> np.array:
    current_utility_row = np.dot(row_strategy, np.dot(matrix, column_strategy))
    current_utility_column = np.dot(column_strategy.T, np.dot(matrix.T, row_strategy.T))

    #best response utility the row player
    best_response_utilities_row = np.dot(matrix, column_strategy)
    best_response_utility_row = np.max(best_response_utilities_row)

    #best response utility column player
    best_response_utilities_column = np.dot(matrix.T, row_strategy.T)
    best_response_utility_column = np.min(best_response_utilities_column)

    #deltas
    delta_row = best_response_utility_row - current_utility_row
    delta_column = best_response_utility_column - current_utility_column

    return np.array([delta_row, delta_column]).flatten()



def compute_exploitability(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    #player deltas
    deltas = compute_deltas(matrix, row_strategy, column_strategy)

    nash_conv = np.sum(deltas)

    #exploitability
    exploitability = nash_conv / 2

    return exploitability


def compute_epsilon(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    deltas = compute_deltas(matrix, row_strategy, column_strategy)

    epsilon = np.max(deltas)

    return epsilon


def compute_exploitability_zero_sum(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    #best response row player
    best_response_utilities_row = np.dot(matrix, column_strategy)
    best_response_utility_row = np.max(best_response_utilities_row)

    #current utility row player
    current_utility_row = np.dot(row_strategy, np.dot(matrix, column_strategy))

    #exploitability is the improvement the row player can make
    exploitability = best_response_utility_row - current_utility_row

    return exploitability



def best_response(matrix, opponent_strategy):
    #best response to an opponent's strategy
    utilities = matrix @ opponent_strategy
    return np.argmax(utilities)

def update_average_strategy(average_strategy, new_strategy, iteration):
    #Update the average strategy
    return (average_strategy * iteration + new_strategy) / (iteration + 1)


def regret_self_play(matrix_player_1,matrix_player_2, num_iterations):
    num_strategies = matrix_player_1.shape[0]
    
    row_regret = np.ones(num_strategies)
    column_regret = np.ones(num_strategies)

    exploitabilities = []

    for i in range(num_iterations):
        #Normalize regrets
        row_regret_normalized  = [float(i)/sum(row_regret) for i in row_regret]
        column_regret_normalized = [float(i)/sum(column_regret) for i in column_regret]
        
        #row deviating vectors
        row_deviating_vector = np.dot(matrix_player_1,column_regret_normalized)
        column_deviating_vector = np.dot(matrix_player_2,row_regret_normalized)
        
        #expected utility
        row_expected_utility = np.dot(row_deviating_vector, row_regret_normalized)
        column_expected_utility = np.dot(column_deviating_vector,column_regret_normalized)

        #regret gains
        row_regret_gain = np.zeros(num_strategies)
        column_regret_gain = np.zeros(num_strategies)
        for i in range(len(row_deviating_vector)):
            strategy_row = row_deviating_vector[i]
            dif = strategy_row - row_expected_utility
            row_regret_gain[i] = max(0, dif)
            
        for i in range(len(column_deviating_vector)):
            strategy_col = column_deviating_vector[i]
            dif = strategy_col - column_expected_utility
            column_regret_gain[i] = max(0, dif)

        #Update regrets
        row_regret = row_regret + row_regret_gain
        column_regret = column_regret + column_regret_gain

        #exploitability of the average strategy
        exploitability = compute_exploitability_zero_sum(matrix_player_1, row_regret_normalized, column_regret_gain)
        #exploitability = compute_exploitability_zero_sum(matrix_player_1, row_regret_gain, column_regret_normalized)
        exploitabilities.append(exploitability)

    return row_regret_normalized, column_regret_normalized, exploitabilities

#regret matching self-play
matrix1 = np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]])
matrix2 = np.array([[0, -1, 1], [1, 0, -1], [2, -1, 0]])
row_regret, column_regret, exp = regret_self_play(matrix1,matrix2, 1000)
print(row_regret)
print(column_regret)
plt.plot(exp)
plt.show()

