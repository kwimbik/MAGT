from pulp import LpMaximize, LpProblem, LpVariable, LpStatus
import numpy as np

def find_nash_equilibrium_lp(payoff_matrix):
    model = LpProblem(name="Nash_Equilibrium", sense=LpMaximize)

    # Strategie hracu
    m = len(payoff_matrix)
    n = len(payoff_matrix[0])

    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1) for i in range(m)]
    y = [LpVariable(f"y_{j}", lowBound=0, upBound=1) for j in range(n)]

    # payoff var
    u = LpVariable("u", lowBound=None)

    # Objective function
    model += u

    # Constrainty
    for j in range(n):
        model += sum(x[i] * payoff_matrix[i][j] for i in range(m)) >= u
    model += sum(x) == 1 

    # Constrainty column player
    for i in range(m):
        model += sum(y[j] * (-payoff_matrix[i][j]) for j in range(n)) <= u
    model += sum(y) == 1

    model.solve()

    # Reseni
    strategy_row_player = [x[i].varValue for i in range(m)]
    strategy_col_player = [y[j].varValue for j in range(n)]
    equilibrium_value = u.varValue

    return strategy_row_player, strategy_col_player, equilibrium_value


def find_correlated_equilibrium_lp_zerosum(payoff_matrix):
    model = LpProblem(name="Correlated_Equilibrium", sense=LpMaximize)

    # Strategie hracu
    m = len(payoff_matrix)
    n = len(payoff_matrix[0])

    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1) for i in range(m)]
    y = [LpVariable(f"y_{j}", lowBound=0, upBound=1) for j in range(n)]

    #  joint probabs vars
    p = [[LpVariable(f"p_{i}_{j}", lowBound=0, upBound=1) for j in range(n)] for i in range(m)]

    # paoff var
    u = LpVariable("u", lowBound=None)

    # Objective function
    model += u

    # Constrainty row player
    for i in range(m):
        model += sum(p[i][j] for j in range(n)) == x[i]  #marginal probabconstraints

    # Constrainty column player
    for j in range(n):
        model += sum(p[i][j] for i in range(m)) == y[j]  #Marginal probab constraints

    model += sum(x) == 1
    model += sum(y) == 1

    model.solve()

    #Reseni
    strategy_row_player = [x[i].varValue for i in range(m)]
    strategy_col_player = [y[j].varValue for j in range(n)]
    joint_probabilities = [[p[i][j].varValue for j in range(n)] for i in range(m)]
    equilibrium_value = u.varValue

    return strategy_row_player, strategy_col_player, equilibrium_value

payoff_matrix = [
    [1, -2, 3],
    [0, -1, 2],
    [-3, 2, -1]
]
# Find Nash equilibrium using LP
nash_equilibrium = find_nash_equilibrium_lp(payoff_matrix)
print("Payoff Matrix:")
for row in payoff_matrix:
    print(row)
print("\nNash Equilibrium:")
print("Row Player Strategy:", nash_equilibrium[0])
print("Column Player Strategy:", nash_equilibrium[1])
print("Equilibrium Value (u):", nash_equilibrium[2])

########################
print("Correalted equilibria")

player_1_matrix = [
    [1, -2, 3],
    [0, -1, 2],
    [-3, 2, -1]
]


correlated_equilibrium = find_correlated_equilibrium_lp_zerosum(player_1_matrix)
print("Payoff Matrix:")
for row in payoff_matrix:
    print(row)
print("\nCorrelated Equilibrium:")
print("Row Player Strategy:", correlated_equilibrium[0])
print("Column Player Strategy:", correlated_equilibrium[1])
print("Equilibrium Value (u):", correlated_equilibrium[2])


