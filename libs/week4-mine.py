from pulp import LpMaximize, LpProblem, LpVariable, LpStatus

def find_nash_equilibrium_lp(payoff_matrix):
    # Define the LP problem
    model = LpProblem(name="Nash_Equilibrium", sense=LpMaximize)

    # Number of strategies for each player
    m = len(payoff_matrix)
    n = len(payoff_matrix[0])

    # Variables representing the probability of each strategy
    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1) for i in range(m)]
    y = [LpVariable(f"y_{j}", lowBound=0, upBound=1) for j in range(n)]

    # Variable representing the minimum expected payoff (u)
    u = LpVariable("u", lowBound=None)

    # Objective function: maximize the minimum expected payoff
    model += u

    # Constraints for row player
    for j in range(n):
        model += sum(x[i] * payoff_matrix[i][j] for i in range(m)) >= u
    model += sum(x) == 1  # Sum of probabilities for row player is 1

    # Constraints for column player
    for i in range(m):
        model += sum(y[j] * (-payoff_matrix[i][j]) for j in range(n)) <= u
    model += sum(y) == 1  # Sum of probabilities for column player is 1

    # Solve the linear program
    model.solve()

    # Extract the solution
    strategy_row_player = [x[i].varValue for i in range(m)]
    strategy_col_player = [y[j].varValue for j in range(n)]
    equilibrium_value = u.varValue

    return strategy_row_player, strategy_col_player, equilibrium_value


def find_correlated_equilibrium_lp_zerosum(payoff_matrix):
    # Define the LP problem
    model = LpProblem(name="Correlated_Equilibrium", sense=LpMaximize)

    # Number of strategies for each player
    m = len(payoff_matrix)
    n = len(payoff_matrix[0])

    # Variables representing the probability of each strategy for individual players
    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1) for i in range(m)]
    y = [LpVariable(f"y_{j}", lowBound=0, upBound=1) for j in range(n)]

    # Variables representing joint probabilities
    p = [[LpVariable(f"p_{i}_{j}", lowBound=0, upBound=1) for j in range(n)] for i in range(m)]

    # Variable representing the minimum expected payoff (u)
    u = LpVariable("u", lowBound=None)

    # Objective function: maximize the minimum expected payoff
    model += u

    # todo fix this
    # Constraints for row player
    # Constraints for row player
    for i in range(m):
        model += sum(p[i][j] for j in range(n)) == x[i]  # Marginal probability constraints
        model += sum(p[i][j] * payoff_matrix[i][j] for j in range(n)) >= u

    # Constraints for column player
    for j in range(n):
        model += sum(p[i][j] for i in range(m)) == y[j]  # Marginal probability constraints
        model += sum(p[i][j] * (-payoff_matrix[i][j]) for i in range(m)) <= u
    # konec todo

    # Probabilities sum to 1
    model += sum(x) == 1
    model += sum(y) == 1

    # Solve the linear program
    model.solve()

    # Extract the solution
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


