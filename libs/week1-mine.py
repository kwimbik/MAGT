import numpy as np
import math




def evaluate(matrix,row_strategy,column_strategy):
    return np.dot(np.dot(row_strategy, matrix),column_strategy)

def best_response_value_row(matrix,row_strategy):
    br = math.inf
    for j in range(len(row_strategy[0])):
        brr = 0
        for i in range(len(row_strategy[0])):
            brr += matrix[i,j] * row_strategy[0,i]
        if brr < br:
            br = brr
    return br

def best_response_value_column(matrix,column_strategy):
    br = - math.inf
    for i in range(len(column_strategy)):
        brc = 0
        for j in range(len(column_strategy)):
            brc +=(matrix[i,j] * column_strategy[j])
        if brc > br:
            br = brc
    return br


def findDominatedStratefies(matrix):
    n = matrix.shape[0]
    m = matrix.shape[1]
    dominated_row = -1
    dominated_col = -1
    # Find dominant strategies for rows
    dominated = False
    for i in range(n):
        for j in range(n):
            if all(matrix[i, k] >= matrix[j, k] for k in range(m)) and any(matrix[i, k] > matrix[j, k] for k in range(m)):
                dominated_row = i
                dominated = True
                break

    # Find dominant strategies for columns
    for j in range(m):
        for i in range(m):
            if all(matrix[k, j] <= matrix[k, i] for k in range(n)) and any(matrix[k, j] < matrix[k, i] for k in range(n)):
                dominated_col = j
                dominated = True
                break
    return dominated,dominated_row,dominated_col


def dominateStrategiesReduction(matrix):
    n = matrix.shape[0]
    m = matrix.shape[1]
    rows= list(range(0,n))
    cols = list(range(0,m))
    dominated = True
    while dominated:
        dominated, dominated_row,dominated_col = findDominatedStratefies(matrix)
        if dominated == False: 
             break
        n = matrix.shape[0]
        m = matrix.shape[1]
        rows= list(range(0,n))
        cols = list(range(0,m))
        rows_to_keep  = [item for item in rows if item is not dominated_row]
        cols_to_keep = [item for item in cols if item is not dominated_col]
        matrix = matrix[np.ix_(rows_to_keep, cols_to_keep)]
    
    #matrix without dominated strategies
    return matrix

matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
row_strategy = np.array([[0.1, 0.2, 0.7]])
column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()


row_value = evaluate(matrix=matrix, row_strategy=row_strategy, column_strategy=column_strategy)
br_value_row = best_response_value_row(matrix, row_strategy)
br_value_column = best_response_value_column(matrix,column_strategy)

print("row_value:", row_value)
print("br_value_row:", br_value_row)
print("br_value_column:", br_value_column)

matrix_dominated = np.array([[1, 2, 1, -1],
                   [3, 1, 3, 4],
                   [1, 3, 3, 6],
                   [2, 2, 3, 0]])

reduced_matrix = dominateStrategiesReduction(matrix_dominated)
print(reduced_matrix)
