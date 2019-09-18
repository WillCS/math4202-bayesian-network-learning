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

    # Cant have a thing be it's own parent
    # Linear variables because we really only care about the linear relaxation
    I = { (W, u): model.addVar()
            for W in parent_sets
            for u in variables if (not (u in W))
    }

    model.setObjective(quicksum(
            score[W, u] * I[W, u]
            for W in parent_sets for u in variables if (W, u) in I
    ))

    # Only one parent set
    convexity_constraints = { u:
        model.addConstr(quicksum(I[W, u] for W in parent_sets if (W, u) in I) == 1)
        for u in variables
    }
        
    # Need one var with no parent for DAG
    sink_constraint = model.addConstr(quicksum(I[emptyset, u] for u in variables) >= 0.99)
    # ^ We'll replace this with the delayed cluster constraint generation

    model.optimize()
    
    result = [(W,u) for W in parent_sets for u in variables if (W, u) in I and I[W, u].x > 0]
    result.sort(key = lambda x:x[1])

    print(find_cutting_plane(variables, parent_sets, result))

    print_parent_visualisation(result)

def find_cutting_plane(variable_range, parent_sets, solution_set):
    cutting_plane_model: Model = Model('Cutting Plane')
    parent_set_range = range(len(parent_sets))

    def vertex_in_cluster(v, cluster) -> int:
        return 1 if v in cluster else 0

    def intersects(W, cluster) -> int:
        return 1 if len(v for v in cluster if v in W) == 0 else 0

    def in_sols(W, u) -> int:
        return 1 if (W, u) in solution_set else 0

    J = { (W, u): cutting_plane_model.addVar(vtype = GRB.BINARY)
        for W in parent_set_range
        for u in variable_range if (not (u in parent_sets[W]))
    }

    cutting_plane_model.setObjective(quicksum(
        in_sols(W, u) * J[W, u]
        for W in parent_set_range
        for u in variable_range if (W, u) in J
    ), GRB.MAXIMIZE)

    cutting_plane_model.addConstr(quicksum(
        in_sols(W, u) * J[W, u]
        for W in parent_set_range
        for u in variable_range if (W, u) in J
    ) <= 0.9999999999)
    
    cluster_constraints = { (W, u): cutting_plane_model.addConstr(
            J[W, u] == 1 - quicksum(J[W_prime, u_prime] 
            for W_prime in parent_set_range
            for u_prime in parent_sets[W] if (W_prime, u_prime) in J
        )) for u in variable_range for W in parent_set_range if (W, u) in J
    }

    cutting_plane_model.optimize()

    print(cutting_plane_model.feasibility())
    exit()

    new_cluster = [u for u in variable_range if
            sum(J[W, u].x for W in parent_set_range if (W, u) in J) > 0]

    return quicksum(I[W, u] for W in parent_set_range for u in variable_range if (W, u) in I) >= 1

# def score_parents(parent_sets, variable_set, scoring_type='RANDOM'):  
#     if scoring_type == 'VALUE':
#         score = {(W, u): u for W in parent_sets for u in variable_set}
#     else:
#         score = {(W, u): random.random() for W in parent_sets for u in variable_set}
#     return score

def print_parent_visualisation(res):
    for parent_child_set in res:
        parents = parent_child_set[0]
        child = parent_child_set[1]
        print(child, '<-', *parents)

solve(dataset, 2)