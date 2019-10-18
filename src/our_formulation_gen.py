from gurobipy import *

import os
import random
import numpy as np
import heapq
import itertools

# from scoring import bdeu_scores,bdeu_scores_sig
from data import Dataset, parse_dataset
from math_utils import binomial_coefficient, factorial, get_subsets_of_size, parse_number


random.seed(13)
dataset_name = "mildew_100.data"

dataset = parse_dataset(dataset_name)

def solve(data: Dataset, parent_set_lim: int, col = False):
    variables = range(data.num_variables)
    num_parent_sets = binomial_coefficient(data.num_variables, parent_set_lim)
    parent_sets = [s for s in get_subsets_of_size(variables, parent_set_lim)]
    emptyset = parent_sets[0]
    
    nodes = optimal_extend_path(variables,{},data)

    #score = score_parents(parent_sets, variables)
    I = []
    parent_sets = [()]
    for var in variables:
        if nodes[var]:
            combo = itertools.combinations(nodes[var],1)
            for i in range(2,len(nodes[var])+1):
                combo = itertools.chain(combo,itertools.combinations(nodes[var],i))
            for x in combo:
                I.append((x,var))
                if not x in parent_sets:
                    parent_sets.append(x)
        else:
           I.append(((),var)) 
    #score = bdeu_scores(data, variables, parent_sets)
    score = bdeu_scores(data, I)
    print(score)

    model = Model('Bayesian Network Learning')

    # Linear variables because we really only care about the linear relaxation
    I = { (W, u): model.addVar()
#            for W in parent_sets
#            for u in variables
            for (W,u) in I
    }

    model.setObjective(quicksum(score[W, u] * I[W, u]
            for (W,u) in I.keys()
    ),GRB.MAXIMIZE)

    # Only one parent set
    convexity_constraints = { u:
            model.addConstr(quicksum(I[W, u] for W in parent_sets if (W,u) in I) == 1)
            for u in variables
    }
        
    model.ModelSense = -1      # maximise objective
    model.Params.PreCrush = 1  # since (always) adding cuts
    model.Params.CutPasses = 100000    # want to allow many cuts
    model.Params.GomoryPasses = 100000 # want to allow many cuts
    model.Params.MIPFocus = 2          # focus on proving optimality
    model.Params.OutputFlag = 0
        
    
    result = {}
    last_obj_value = 0
    cluster = []
#    while True:
    plane = Model('Cutting Plane')
    i = 0
    exit1 = False
    while True:
        while True:
            print(i)
            try:
                last_graph = set([(u,W) for (W,u) in I.keys() if I[W,u].x > 0.0001])
            except:
                last_graph = set()
            model.reset()
            print("here 1")
            model.optimize()

                
    
    
            last_obj_value = model.objVal
    
            if model.status == GRB.Status.INFEASIBLE:
                print("failed")
                exit(0)
            x = set([(u,W) for (W,u) in I.keys() if I[W,u].x > 0.0001]).difference(last_graph)
            print(x)
            if not x:
                result = {}
                result = { (W, u): I[W, u].x 
                    for (W,u) in I.keys()
                }
                result = [(W,u) for (W,u) in result.keys() if result[W,u] > 0.001]
                print_parent_visualisation(result)
                print(model.objVal)
                return
            
            
            
            result = {}
            result = { (W, u): I[W, u].x 
                    for (W,u) in I.keys()
            }
            print("here 2")
            new_cluster = find_cluster(variables, parent_sets, result)
            
            print(new_cluster)
            print("here 3")
            i += 1
            for x in new_cluster:
                cluster.append(x)
                model.addLConstr(quicksum(I[W, u] for u in x for W in parent_sets if intersection_size(W,x) < 1), GRB.GREATER_EQUAL, 1)
                model.addLConstr(quicksum(I[W, u] for u in x for W in parent_sets if intersection_size(W,x) < 2), GRB.GREATER_EQUAL, 2)
    if col:
        result = {}
        result = { (W, u): I[W, u].x 
            for (W,u) in I.keys()
        }
        result = [(W,u) for (W,u) in result.keys() if result[W,u] > 0.001]
        print_parent_visualisation(result)
        print(model.objVal)
        return
        variables_set = set()
        for x in variables:
            variables_set.add(x)
        for x in result:
            added = extend_path(x,variables_set,score)
            for y in added:
                    score_parents([y[0]],[y[1]],scordict=score)
                    if not y in I:
                        I[y]  = model.addVar(obj = 1)
                    else:
                        print(y)
        model.setObjective(quicksum(score[W, u] * I[W, u]
                for (W,u) in I.keys()
        ),GRB.MAXIMIZE)
    
        convexity_constraints = { u:
                model.addConstr(quicksum(I[W, u] for W in parent_sets if (W,u) in I) == 1)
                for u in variables
        }
        for x in cluster:
            model.addLConstr(quicksum(I[W, u] for u in x for W in parent_sets if intersection_size(W,x) < 1), GRB.GREATER_EQUAL, 1)
            model.addLConstr(quicksum(I[W, u] for u in x for W in parent_sets if intersection_size(W,x) < 2), GRB.GREATER_EQUAL, 2)
        
def optimal_extend_path(variables,score,data):
    variables = set(variables)
    i = 0
    added = {}
    for x in variables:
        added[x] = set()
    for i in range(4):
        for x in variables:
            test = None
            print("var")
            print(x)
            parents = added[x]
            if len(parents) == i:
                for y in variables.difference(parents).difference(set([x])):
                    if distence(x, parents.union(set([y])),data,score) < distence(x, parents,data,score):
                        if test:
                            if distence(x, parents.union(set([y])),data,score) < distence(x, test,data,score):
                                test = parents.union(set([y]))
                        else:
                            test = parents.union(set([y]))
                if test:
                    parents = test
                    print(parents)
                    added[x] = parents
                
#        expaend = False
#        while not expaend:
#            print(x)
#            parents = set([x])
#            expaend = False
#            for y in variables.difference(parents):
#                if distence(y, parents,data,score) == distence(y,variables.difference(set([y])), data,score):
#                    print("added")
#                    parents = parents.union(set([y]))
#                    expaend = True
#        added[x] = parents.difference(set([x]))
#        print("done")
    return added

def distence(var, parents, data,scoredict):
    if not parents or len(parents) <= 1:
        if (var,tuple(parents)) in scoredict:
            score = scoredict[var,tuple(parents)]
        else:
            score = bdeu_scores_sig(data,var,tuple(parents))
            scoredict[var,tuple(parents)] = score
        return score
    if (var,tuple(parents)) in scoredict:
        score = scoredict[var,tuple(parents)]
    else:
        score = bdeu_scores_sig(data,var,tuple(parents))
        scoredict[var,tuple(parents)] = score
    other = []
    for x in parents:
        other.append(distence(var,[x],data,scoredict))
        
    else:
        other = [1000000000000000000]
    other = min(other)
    return min(score, other)
    

def extend_path(nodeset,varset,score):
    parents = nodeset[0]
    nodese = set(nodeset[0])
    nodeset = (nodese,nodeset[1])
    expend = True
    size = 2
    added = set()
    nodes = varset.difference(nodeset[0])
    if nodeset[1] in nodes:
        nodes.remove(nodeset[1])
    i = 0
    while expend:
        expend = False
        size += 1
        for x in itertools.combinations(varset,3):
            i += 1
            if d(x,parents,score) <= d(x,tuple(varset.difference(x)),score):
                this = set(parents)
                this = this.union(set(x))
#                print(d(x,parents,score))
#                print(d(x,tuple(varset.difference(x)),score))
                #print(this)
                parents = tuple(sorted(tuple(this)))
                expend = True
                added.add((parents,nodeset[1]))
        if len(added) >=  10:
            return added
    return added
                

def d(node1,node2,scoredict):
    if not node2:
        return -100000
    if (node1,node2) in scoredict:
        score = scoredict[node1,node2]
    else:
        scoredict = score_parents([node1],[node2],scordict=scoredict)
        score = scoredict[node1,node2]
    for x in node2:
        if not (node1,x) in scoredict:
            scoredict = score_parents([node1],[x],scordict=scoredict)
    other = min([scoredict[node1,x] for x in node2])
    other = random.random()
    return min(score, other)


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
        if x in var_set:
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

#     J(W -> u) variables for sub problem
    J = { (W, u): cutting_plane_model.addVar(vtype = GRB.BINARY)
        for (W,u) in solution_set.keys() if solution_set[W,u] > 0.01 
    }
    
         
    K = { (u): cutting_plane_model.addVar(vtype = GRB.BINARY)
        for u in variable_range
    }
    
    ksum = cutting_plane_model.addVar()
    cutting_plane_model.addConstr(ksum == quicksum(K.values()))
    ksum.BranchPriority = 100
    for u in variable_range:
        K[u].BranchPriority = 10

    cutting_plane_model.setObjective(quicksum(
        solution_set[W,u]*J[W, u]
        for (W,u) in J.keys()
    )-quicksum(K[u] for u in variable_range), GRB.MINIMIZE)
#
    # Objective value must be strictly less than 1
    cutting_plane_model.addConstr(quicksum(
        solution_set[W, u] * J[W, u]
        for (W,u) in J.keys()
    )-quicksum(K[u] for u in variable_range) >= -0.98)

#    # These constraints come from (8) in the paper
    acyclicity_constraints = { (W, u): cutting_plane_model.addLConstr(
            (1-J[W, u]) + quicksum(K[x] for x in W) >= 1 )
            for (W,u) in J.keys()
    }
    
    acyclicity_constraints2 = { (W, u): cutting_plane_model.addLConstr(
            (1-J[W, u]) + K[u] >= 1 )
            for (W,u) in J.keys()
    }
  
#    
    cutting_plane_model.addLConstr(
            quicksum(K[u] for u in variable_range) >= 2)

    cutting_plane_model.Params.OutputFlag = 0
    cutting_plane_model.setParam('GURO_PAR_MINBPFORBID', 1)
    cutting_plane_model.ModelSense = -1
    cutting_plane_model.Params.PoolSolutions = 200
    cutting_plane_model.Params.PoolSearchMode = 1

    cutting_plane_model.optimize()
    nsols = cutting_plane_model.Solcount 
    cluster = set()
    for i in range(nsols):
        cutting_plane_model.Params.SolutionNumber = i
        new_cluster = tuple([u for u in variable_range if K[u].x > 0.5])
        cluster.add(new_cluster)
    return cluster

def intersects(W, cluster) -> int:
    return 1 if len([v for v in cluster if v in W]) == 0 else 0

def intersection_size(W, cluster) -> int:
    return len([v for v in cluster if v in W])

def score_parents(parent_sets, variable_set, scoring_type='RANDOM',scordict = None):  
    if scordict:
        for W in parent_sets:
            for u in variable_set:
                scordict[W,u] = random.random()
        return scordict
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