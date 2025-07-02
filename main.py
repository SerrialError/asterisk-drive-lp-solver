import pulp
import math


def solve_force_lp(k_min, k_max, theta_Fr, tau_r, L, W):
    """
    Solves the following LP:
      maximize c^T x
      subject to:
        k_min[i] <= x[i] <= k_max[i]
        a_ang^T x = 0
        |a_tau^T x| <= tau_r

    Parameters:
        k_min     : list or tuple of length 6, minimum force values
        k_max     : list or tuple of length 6, maximum force values
        theta_Fr  : float, angle theta_Fr in radians
        tau_r     : float, allowable moment (scalar)
        L, W      : floats, geometry parameters

    Returns:
        dict with variable names and their optimal values, and objective
    """
    # Decision variable names
    x_names = ['F_m1', 'F_m2', 'F_m3', 'F_m4', 'F_o1', 'F_o2']

    # Objective coefficients
    c = [0, 2, 2, 0, 1, 1]

    # Define LP problem
    prob = pulp.LpProblem("Force_Optimization", pulp.LpMaximize)

    # Create variables with bounds
    x = {
        name: pulp.LpVariable(name, lowBound=k_min[i], upBound=k_max[i])
        for i, name in enumerate(x_names)
    }

    # Objective
    prob += pulp.lpSum(c[i] * x[x_names[i]] for i in range(6)), "Objective"

    # Angle constraint coefficients: 1+tan, 1-tan, ...
    t = math.tan(theta_Fr)
    a_ang = [1 + t, 1 - t, 1 - t, 1 + t, 1, 1]

    # Enforce a_ang^T x == 0 via two inequalities
    prob += pulp.lpSum(a_ang[i] * x[x_names[i]] for i in range(6)) <= 0, "Angle_pos"
    prob += pulp.lpSum(a_ang[i] * x[x_names[i]] for i in range(6)) >= 0, "Angle_neg"

    # Torque constraint coefficients
    coeff = (L + W) / 4.0
    a_tau = [-coeff, coeff, -coeff, coeff, -W/2.0, W/2.0]

    # Enforce |a_tau^T x| <= tau_r
    prob += pulp.lpSum(a_tau[i] * x[x_names[i]] for i in range(6)) <= tau_r, "Torque_pos"
    prob += pulp.lpSum(a_tau[i] * x[x_names[i]] for i in range(6)) >= -tau_r, "Torque_neg"

    # Solve
    prob.solve()

    # Collect results
    solution = {name: var.varValue for name, var in x.items()}
    solution['Objective'] = pulp.value(prob.objective)
    solution['Status'] = pulp.LpStatus[prob.status]

    return solution


# Example parameters
k_min = [-1, -1, -1, -1, -1, -1]
k_max = [1, 1, 1, 1, 1, 1]
theta_Fr = math.radians(45)  # 45 degrees
tau_r = 0
L = 2.0
W = 1.0

result = solve_force_lp(k_min, k_max, theta_Fr, tau_r, L, W)
print("LP Status:", result['Status'])
print("Optimal Objective:", result['Objective'])
for var in ['F_m1', 'F_m2', 'F_m3', 'F_m4', 'F_o1', 'F_o2']:
    print(f"{var} = {result[var]}")
