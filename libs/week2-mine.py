from typing import List, Optional
from scipy.optimize import linprog
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math



def best_response(matrix, opponent_strategy):
    # best response
    utilities = matrix @ opponent_strategy
    return utilities[np.argmin(utilities)]


def best_response_matrix(matrix, step_size):
    x = 0
    values = []
    while x <= 1:
        strategy = np.array([x, 1-x])
        values.append(best_response(matrix.T, strategy))
        x += step_size
    return values


def verify_support_one_side(matrix: np.array, support_row: List, support_col: List) -> Optional[List]:
    lhs_eq  = []
    rhs_eq = []
    for I in support_row:
        lhs = [ matrix[I,J] for J in  support_col]
        lhs.append(-1)
        lhs_eq.append(lhs)

    one_sum = np.ones(len(support_col)).tolist()
    one_sum.append(0)
    lhs_eq.append(one_sum)
    
    rhs_eq = np.zeros(len(support_row)).tolist()
    rhs_eq.append(1)
        
    c_eq = np.ones(len(support_col) + 1).tolist()
    
    opt = linprog(c=c_eq,b_eq=rhs_eq,
               A_eq = lhs_eq)
    return opt.x

def iterate_Equilibria_equilibria(matrix_p1: np.array,matrix_p2: np.array):
    support_row = matrix_p1.shape[1]
    support_col = matrix_p2.T.shape[1]

    row_ar = [i for i in range(support_row)]
    col_ar = [i for i in range(support_col)]


    rows = sublists(row_ar)
    columns = sublists(col_ar)
    for r in rows:
        if len(r) > 1:
            for c in columns:
                if len(c) > 1:
                    res = find_equilibria(matrix_p1, matrix_p2,r, c)
                    #mame reseni!!
                    if res.x is not None:
                        return [(res.x[:len(c)]),r,c]

    
def sublists(lst):
    n = len(lst)
    sublists = []
     
    for start in range(n):
        for end in range(start + 1, n + 1):
            sublists.append(lst[start:end])
     
    return sublists


def find_equilibria(matrix_p1: np.array,matrix_p2: np.array, support_row: List, support_col: List) -> Optional[List]:
    lhs_eq  = []
    rhs_eq = []
    for I in support_row:
        lhs = [ matrix_p1[I,J] for J in  support_col]
        lhs.append(-1)
        lhs += np.zeros(len(support_row) + 1).tolist()
        lhs_eq.append(lhs)

    one_sum = np.ones(len(support_col)).tolist()
    one_sum.append(0)
    one_sum += np.zeros(len(support_row) + 1).tolist()
    lhs_eq.append(one_sum)
    
    rhs_eq = np.zeros(len(support_row)).tolist()
    rhs_eq.append(1)
    #konec rovnic prvniho hrace, pro druheho musime do rovnic insertovat nuly

    for I in support_col:
        lhs = np.zeros(len(support_col) + 1).tolist()
        lhs += [matrix_p2.T[I,J] for J in  support_row]
        lhs.append(-1)
        lhs_eq.append(lhs)

    one_sum = np.zeros(len(support_col) + 1).tolist()
    one_sum += np.ones(len(support_row)).tolist()
    one_sum.append(0)
    lhs_eq.append(one_sum)

    rhs_eq += np.zeros(len(support_col)).tolist()
    rhs_eq.append(1)
    
    c_eq = np.ones(len(support_col) + len(support_row) + 2).tolist()
    
    opt = linprog(c=c_eq,b_eq=rhs_eq,
               A_eq = lhs_eq)
    
    return opt



matrix_p1 = np.array([[0, 0, -10], [1, -10, -10], [-10, -10, -10]])
matrix_p2 = np.array([[0, 1, -10], [0, -10, -10], [-10, -10, -10]])


#result = find_equilibria(matrix_p1 = matrix_p1,matrix_p2 = matrix_p2, support_row=[0], support_col = [0])
result = iterate_Equilibria_equilibria(matrix_p1,matrix_p2)
print(result)



matrix = np.array([[-1, 0, -0.8], [1, -1, -0.5]]) 
step_size = 0.001
result = best_response_matrix(matrix, step_size)
points = int((1 / step_size))  + 1
x_axis = list(range(1,points))

plt.plot(x_axis, result, 'bo')
plt.show()
