import numpy as np


def detect_basic_variables(tableau, variable_names):
    num_constraints = tableau.shape[0] - 1

    basic_variables = []

    for row in range(num_constraints):
        basic_variable = None

        for col in range(len(variable_names)):

            column = tableau[:, col]
            constraint_column = column[:-1]

            ones = np.isclose(constraint_column, 1)
            zeros = np.isclose(constraint_column, 0)

            if np.sum(ones) == 1 and np.sum(~zeros) == 1:
                if ones[row]:
                    basic_variable = variable_names[col]
                    break

        basic_variables.append(basic_variable)

    return basic_variables


def is_optimal(tableau):
    return np.all(tableau[-1, :-1] >= 0)


def get_pivot_column(tableau):
    return int(np.argmin(tableau[-1, :-1]))


def get_pivot_row(tableau, pivot_col, num_constraints):
    ratios = []

    for i in range(num_constraints):
        element = tableau[i, pivot_col]

        if element > 0:
            ratio = tableau[i, -1] / element
        else:
            ratio = np.inf

        ratios.append(ratio)

    return int(np.argmin(ratios))


def get_variable_name(variable_names, index):
    return variable_names[index]


def pivot(tableau, pivot_row, pivot_col):
    pivot_element = tableau[pivot_row, pivot_col]

    tableau[pivot_row] = tableau[pivot_row] / pivot_element

    for i in range(len(tableau)):
        if i != pivot_row:
            factor = tableau[i, pivot_col]
            tableau[i] = tableau[i] - factor * tableau[pivot_row]


def solve_simplex(tableau, variable_names):

    num_constraints = tableau.shape[0] - 1

    basic_variables = detect_basic_variables(tableau, variable_names)

    iterations = []

    # Inicial
    iterations.append({
        "iteration": 0,
        "tableau": tableau.copy(),
        "basic_variables": basic_variables.copy(),
        "entering_variable": None,
        "leaving_variable": None,
        "message": "Tableau inicial"
    })

    while not is_optimal(tableau):

        pivot_col = get_pivot_column(tableau)
        pivot_row = get_pivot_row(tableau, pivot_col, num_constraints)

        entering_variable = get_variable_name(variable_names, pivot_col)
        leaving_variable = basic_variables[pivot_row]

        basic_variables[pivot_row] = entering_variable

        pivot(tableau, pivot_row, pivot_col)

        iterations.append({
            "iteration": len(iterations),
            "tableau": tableau.copy(),
            "basic_variables": basic_variables.copy(),
            "entering_variable": entering_variable,
            "leaving_variable": leaving_variable,
            "message": f"Entra {entering_variable}, sale {leaving_variable}"
        })

    # solución final
    solution = {v: 0 for v in variable_names}

    for i, var in enumerate(basic_variables):
        solution[var] = tableau[i, -1]

    optimal_value = tableau[-1, -1]

    return {
        "solution": solution,
        "optimal_value": optimal_value,
        "tableau": tableau,
        "iterations": iterations,
        "basic_variables": basic_variables
    }