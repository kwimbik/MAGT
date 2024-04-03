import numpy as np
import matplotlib.pyplot as plt
# import libs.week1 as week1


def compute_deltas(matrix: np.array, row_strategy: np.array, column_strategy: np.array):
    current_utility_row = np.dot(row_strategy, np.dot(matrix, column_strategy))
    current_utility_column = np.dot(column_strategy.T, np.dot(matrix.T, row_strategy.T))

    # best response utility for the row player
    best_response_utilities_row = np.dot(matrix, column_strategy)
    best_response_utility_row = np.max(best_response_utilities_row)

    #best response utility for the column player
    best_response_utilities_column = np.dot(matrix.T, row_strategy.T)
    best_response_utility_column = np.min(best_response_utilities_column)

    # deltas
    delta_row = best_response_utility_row - current_utility_row
    delta_column = best_response_utility_column - current_utility_column

    return np.array([delta_row, -delta_column]).flatten()



def compute_exploitability(matrix: np.array, row_strategy: np.array, column_strategy: np.array):
        # Compute the deltas for both players
    deltas = compute_deltas(matrix, row_strategy, column_strategy)

    # Compute nashovc as the sum of the deltas
    nash_conv = np.sum(deltas)

    exploitability = nash_conv / 2

    return exploitability


def compute_epsilon(matrix: np.array, row_strategy: np.array, column_strategy: np.array):
    deltas = compute_deltas(matrix, row_strategy, column_strategy)

    epsilon = np.max(deltas)

    return epsilon


def compute_exploitability_zero_sum(matrix: np.array, row_strategy: np.array, column_strategy: np.array):
    #best response utility row player
    best_response_utilities_row = np.dot(matrix, column_strategy)
    best_response_utility_row = np.max(best_response_utilities_row)

    #Curr utility for row player
    current_utility_row = np.dot(row_strategy, np.dot(matrix, column_strategy))

    exploitability = best_response_utility_row - current_utility_row

    return exploitability



def best_response(matrix, opponent_strategy):
    # best response
    utilities = matrix @ opponent_strategy
    return np.argmax(utilities)

def best_response_complet(matrix, opponent_strategy):
    # best response
    utilities = matrix @ opponent_strategy
    return utilities

def update_average_strategy(average_strategy, new_strategy, iteration):
    #updet the average strategy
    return (average_strategy * iteration + new_strategy) / (iteration + 1)


def naive_self_play(matrix, num_iterations):
    num_strategies = matrix.shape[0]
    row_strategy = np.zeros(num_strategies)
    column_strategy = np.zeros(num_strategies)

    row_average_strategy = np.zeros(num_strategies)
    column_average_strategy = np.zeros(num_strategies)

    exploitabilities = []
    exploitabilities_avg = []

    for i in range(num_iterations):
        

        
        row_strategy_old = row_strategy
        # Row player response
        row_response = best_response(matrix, column_strategy)
        #row_response = best_response(matrix, column_strategy_average)
        row_strategy[:] = 0
        row_strategy[row_response] = 1


        # Column player response
        column_response = best_response(-matrix.T, row_strategy_old)
        #column_response = best_response(-matrix.T, row_strategy_average)
        column_strategy[:] = 0
        column_strategy[column_response] = 1


        # update average startegies
        row_average_strategy = update_average_strategy(row_average_strategy, row_strategy, i)
        column_average_strategy = update_average_strategy(column_average_strategy, column_strategy, i)

        #exploitability of the strategy
        exploitability = compute_exploitability_zero_sum(matrix, row_average_strategy, column_average_strategy)
        exploitabilities.append(exploitability)


    return row_average_strategy, column_average_strategy, exploitabilities

#Deltas
matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
row_strategy = np.array([[0.1, 0.2, 0.7]])
column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()
deltas = compute_deltas(matrix,row_strategy,column_strategy)
print(deltas[0],deltas[1])

# fictitious and naive self-play
a, b, exploitabilities = naive_self_play(matrix, 100)
plt.plot(exploitabilities)
plt.show()
