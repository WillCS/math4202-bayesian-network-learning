from gurobipy import *

import argparse
import random


from scoring import bdeu_scores
from data import Dataset, parse_dataset
from math_utils import binomial_coefficient, factorial, get_subsets_of_size, parse_number

random.seed(13)


def solve(data: Dataset, parent_set_lim: int):
    variables = range(data.num_variables)
    num_parent_sets = binomial_coefficient(data.num_variables, parent_set_lim)
    parent_sets = [s for s in get_subsets_of_size(variables, parent_set_lim)]
    emptyset = parent_sets[0]

    #score = score_parents(parent_sets, variables)
    score = bdeu_scores(data, variables, parent_sets)

    model = Model('Bayesian Network Learning')

    # Linear variables because we really only care about the linear relaxation
    I = { (W, u): model.addVar()
            for W in parent_sets
            for u in variables if not u in W
    }

    model.setObjective(quicksum(score[W, u] * I[W, u]
            for (W,u) in I.keys()
    ))

    # Only one parent set
    convexity_constraints = { u:
            model.addConstr(quicksum(I[W, u] for W in parent_sets if (W,u) in I) == 1)
            for u in variables
    }
    
    result = {}
    last_obj_value = 0
    cluster = {}
    cluster_constrs = {}
#    while True:
    for i in range(1):
        try:
            last_graph = set([(u,W) for (W,u) in I.keys() if I[W,u].x > 0.0001])
        except:
            last_graph = set()
        model.reset()
        model.optimize()
        print(set([(u,W) for (W,u) in I.keys() if I[W,u].x > 0.0001]).difference(last_graph))

#        if abs(last_obj_value - model.objVal) < 0.000000001:
#            break

        last_obj_value = model.objVal

        if model.status == GRB.Status.INFEASIBLE:
            exit(0)

        result = { (W, u): I[W, u].x 
                for (W,u) in I.keys()
        }
        graph = {u:W for (W,u) in result.keys() if result[W,u] > 0.0001}
        if not cycles(graph,variables):
            print(done)
            break

#        print()
#        print(result)
        new_cluster = find_cluster(variables, parent_sets, result)
        
        if tuple(new_cluster) in cluster:
            constr = cluster[tuple(new_cluster)] + 1
            model.addConstr(quicksum(I[W, u] for u in new_cluster for W in parent_sets if intersection_size(W,new_cluster) < 1) >= constr)
            cluster[tuple(new_cluster)] = constr
            print(constr)
            print(new_cluster)
        else:
            model.addConstr(quicksum(I[W, u] for u in new_cluster for W in parent_sets if intersection_size(W,new_cluster) < 1) >= 1)
            print(str(tuple(new_cluster)) + "added")
            cluster[tuple(new_cluster)] = 1
            print(cluster)
            print("now")
        # model.addConstr(quicksum(I[W, u] for u in new_cluster for W in parent_sets if not intersects(W, new_cluster) < 2) >= 2)

        result = { (W, u): I[W, u].x 
            for (W,u) in I.keys()
    }
    result = [(W,u) for (W,u) in result.keys() if result[W,u] > 0.001]
    print_parent_visualisation(result)
    print(model.objVal)




def cycles(graph,variables):
    q = []   
    q.append(variables[0])
    var_set = set(variables)
    seen = set()
    while q:
        x = q.pop(0)
        if x in seen:
            return True
        seen.add(x)
        var_set.remove(x)
        for y in graph[x]:
            if y in seen:
                return True
            q.append(y)
    if var_set:
         return cycles(graph,list(var_set))
    return False


def find_cluster(variable_range, parent_sets, solution_set):
    cutting_plane_model: Model = Model('Cutting Plane')
    parent_set_range = range(len(parent_sets))
    empty_set = ()
    parents_lenght = 3

    # J(W -> u) variables for sub problem
    J = { (W, u): cutting_plane_model.addVar(vtype = GRB.BINARY)
        for W in parent_sets
        for u in variable_range if (W, u) in solution_set
    }
    
    K = { (empty_set, u): cutting_plane_model.addVar(vtype = GRB.BINARY)
        for u in variable_range
    }
    
    K_conts = { (empty_set, u): cutting_plane_model.addConstr(K[empty_set, u] == 1 - J[empty_set, u])
        for u in variable_range
    }
    
    L = { (W, u): cutting_plane_model.addVar(vtype = GRB.BINARY)
        for W in parent_sets
        for u in variable_range if (W, u) in solution_set
    }

    cutting_plane_model.setObjective(quicksum(
        solution_set[W, u] * J[W, u]
        for W in parent_sets
        for u in variable_range if (W, u) in solution_set
    ), GRB.MINIMIZE)

    # Objective value must be strictly less than 1
    cutting_plane_model.addConstr(quicksum(
        solution_set[W, u] * J[W, u]
        for W in parent_sets
        for u in variable_range if (W, u) in solution_set
    ) <= 0.989)

    # These constraints come from (8) in the paper
    acyclicity_constraints = { (W, u): cutting_plane_model.addConstr(
            J[W, u] <= J[empty_set, u] )
            for W in parent_sets
            for u in variable_range if (W, u) in solution_set
    }
    
    acyclicity_constraints2 = { (W, u): cutting_plane_model.addGenConstrAnd( 
            L[W, u], [K[empty_set,u_] for u_ in W],"andconstr")
            for W in parent_sets
            for u in variable_range if (W, u) in solution_set
    }
    
    acyclicity_constraints3 = { (W, u): cutting_plane_model.addConstr(
            J[W, u] <= L[W, u] )
            for W in parent_sets
            for u in variable_range if (W, u) in solution_set
    }
    
    
    cutting_plane_model.addConstr(
            quicksum(J[empty_set, u] for u in variable_range) >= 2 )

    cutting_plane_model.optimize()

    new_cluster = [u for u in variable_range if J[(), u].x > 0.99]

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


def main(data_dir, parent_limit):
    dataset = parse_dataset(data_dir)
    solve(dataset, parent_limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Paper implementation of exact bayesian network construction via integer programming")
    parser.add_argument("-d", "--datadir", dest="datadir",
                        help="Directory path containing data",
                        metavar="FILE", default='data/Mildew_100.data')
    parser.add_argument("-p", "--parentlimit", dest="parentlimit",
                        help="limit to parent set size",
                        metavar="INT", default=2)

    args = parser.parse_args()
    main(args.datadir, int(args.parentlimit))