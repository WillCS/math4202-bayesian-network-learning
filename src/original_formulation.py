from gurobipy import *

import os
import random
import numpy as np

from scoring import bdeu_scores
from data import Dataset, parse_dataset
from math_utils import binomial_coefficient, factorial, get_subsets_of_size, parse_number

random.seed(13)

dataset_name = sys.argv[1]

dataset = parse_dataset(dataset_name)

def solve(data: Dataset, parent_set_lim: int):
    variables = range(data.num_variables)
    num_parent_sets = binomial_coefficient(data.num_variables, parent_set_lim)
    parent_sets = [s for s in get_subsets_of_size(variables, parent_set_lim)]
    emptyset = parent_sets[0]

    # score = score_parents(parent_sets, variables)
    score = bdeu_scores(data, variables, parent_sets)

    model = Model('Bayesian Network Learning')

    # Linear variables because we really only care about the linear relaxation
    I = { (W, u): model.addVar()
            for W in parent_sets
            for u in variables
    }

    model.setObjective(quicksum(score[W, u] * I[W, u]
            for W in parent_sets 
            for u in variables
    ))

    # Only one parent set
    convexity_constraints = { u:
            model.addConstr(quicksum(I[W, u] for W in parent_sets) == 1)
            for u in variables
    }
    
    result = {}
    last_obj_value = 0

    while True:
        model.optimize()

        if abs(last_obj_value - model.objVal) < 0.000000001:
            break

        last_obj_value = model.objVal

        if model.status == GRB.Status.INFEASIBLE:
            exit(0)

        result = { (W, u): I[W, u].x 
                for W in parent_sets
                for u in variables
        }

        new_cluster = find_cluster(variables, parent_sets, result)

        model.addConstr(quicksum(I[W, u] for u in new_cluster for W in parent_sets if intersection_size(W, new_cluster) < 1) >= 1)

        # model.addConstr(quicksum(I[W, u] for u in new_cluster for W in parent_sets if not intersects(W, new_cluster) < 2) >= 2)


    print_parent_visualisation(result)

def find_cluster(variable_range, parent_sets, solution_set):
    cutting_plane_model: Model = Model('Cutting Plane')
    parent_set_range = range(len(parent_sets))
    empty_set = parent_sets[0]

    # J(W -> u) variables for sub problem
    J = { (W, u): cutting_plane_model.addVar(vtype = GRB.BINARY)
        for W in parent_sets
        for u in variable_range if (W, u) in solution_set
    }

    cutting_plane_model.setObjective(quicksum(
        solution_set[W, u] * J[W, u]
        for W in parent_sets
        for u in variable_range if (W, u) in J
    ), GRB.MAXIMIZE)

    # Objective value must be strictly less than 1
    cutting_plane_model.addConstr(quicksum(
        solution_set[W, u] * J[W, u]
        for W in parent_sets
        for u in variable_range if (W, u) in J
    ) <= 0.9999999999)
    

    # If J[empty_set, u] == 1 then u is in the cluster.
    # These constraints come from (8) in the paper
    acyclicity_constraints = { (W, u): cutting_plane_model.addConstr(
            J[W, u] == J[W, u] * (J[empty_set, u] + quicksum(J[empty_set, u_prime] for u_prime in W)))
            for W in parent_sets
            for u in variable_range if (W, u) in J
    }

    cutting_plane_model.optimize()

    new_cluster = [u for u in variable_range if
            sum(J[W, u].x for W in parent_sets if (W, u) in J) > 0]

    return new_cluster

def intersects(W, cluster) -> int:
    return 1 if len([v for v in cluster if v in W]) == 0 else 0

def intersection_size(W, cluster) -> int:
    return len([v for v in cluster if v in W])

def score_parents(parent_sets, variable_set, scoring_type='RANDOM'):  
    if scoring_type == 'VALUE':
        score = {(W, u): u for W in parent_sets for u in variable_set}
    else:
        score = {(W, u): random.random() for W in parent_sets for u in variable_set}
    return score

def print_parent_visualisation(res):
    for parent_child_set in res:
        parents = parent_child_set[0]
        child = parent_child_set[1]
        print(child, '<-', *parents)

solve(dataset, 2)