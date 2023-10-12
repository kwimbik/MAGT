import numpy as np
import math

matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
row_strategy = np.array([[0.1, 0.2, 0.7]])
column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()





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
            brc += matrix[i,j] * column_strategy[j]
            print(matrix[i,j],column_strategy[j])
        if brc > br:
            br = brc
    return br

row_value = evaluate(matrix=matrix, row_strategy=row_strategy, column_strategy=column_strategy)
br_value_row = best_response_value_row(matrix, row_strategy)
br_value_column = best_response_value_column(matrix,column_strategy)

print("row_value:", row_value)
print("br_value_row:", br_value_row)
print("br_value_column:", br_value_column)
