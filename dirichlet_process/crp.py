# Generate table assignments for `num_customers` customers, according to
# a Chinese Restaurant Process with dispersion parameter `alpha`.
#
# returns an array of integer table assignments

import numpy as np
from scipy.stats import dirichlet

def stats(scale_factor, G0=[0.2, 0.2, 0.6], N=10000):
    samples = dirichlet(scale_factor*np.array(G0))

def chinese_restaurant_process(num_customers, alpha):
    if num_customers <= 0:
        return []

    table_assignment = [1]
    next_open_table = 2
    for i in range(1, num_customers):
        if np.random.rand(1) < alpha*1.0/(i+alpha):
            table_assignment.append(next_open_table)
            next_open_table += 1
        else:
            table = table_assignment[np.random.randint(len(table_assignment))]
            table_assignment.append(table)
    print table_assignment
    return table_assignment


def polya_urn_model(num_balls, alpha):
    if num_balls <= 0:
        return []
    balls_in_urn = [round(np.random.randn(1), 2)]
    for i in range(1, num_balls):
        if np.random.rand(1) < alpha*1.0/(i + alpha):
            new_color = round(np.random.randn(1), 2)
            balls_in_urn.append(float(new_color))
        else:
            ball = balls_in_urn[np.random.randint(len(balls_in_urn))]
            balls_in_urn.append(ball)
    print balls_in_urn
    return balls_in_urn


if __name__ == '__main__':
    # chinese_restaurant_process(10, 5)
    polya_urn_model(10, 0.3)
