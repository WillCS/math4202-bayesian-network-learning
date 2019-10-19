from gurobipy import *
import argparse
from scoring.scoring import bdeu_scores, bdeu_scores_sig, score_parents
from data import Dataset, parse_dataset
from timeit import default_timer as timer
from math_utils import binomial_coefficient, factorial, get_subsets_of_size, parse_number
import random


class Solver:
    """
    Problem solver class for the MIP.
    """
    def __init__(self, data, parent_set_limit, approach='branching'):
        # General data for the problem 
        self.dataset = data
        self.parent_lim = parent_set_limit
        genthereway = False
        genourway = False
        self.parent_sets = [s for s in get_subsets_of_size(data.num_variables, self.parent_lim)]
        self.scores = bdeu_scores(self.dataset, self.parent_sets)
        print(self.scores[(),1])
        self.callback = True

        # Problem configs 
        self.solver_approach = approach
        
        # Models
        self.master_problem_model = self.get_master_model('Bayesian Network Learning')
        model = self.master_problem_model
        self.cutting_plane_models = {}

        # Master problem constraints
        self.master_convexity_constraints = {}

        # Cutting plane constraints
        self.clusters = []
        self.num_clusters = 0
        self.cluster_constr_k1 = {}
        self.cluster_constr_k2 = {}

        # track callback no.
        self.callback_no = 0
        self.c = 0

    @staticmethod
    def get_master_model(title):
        model = Model(title)

        if not verbose:
            model.Params.OutputFlag = 0        #

        model.ModelSense = -1               # maximise objective
        model.Params.PreCrush = 1           # since (always) adding cuts
        model.Params.CutPasses = 100000     # want to allow many cuts
        model.Params.GomoryPasses = 100000  # want to allow many cuts
        model.Params.MIPFocus = 2           # focus on proving optimality
        model.setParam('LazyConstraints', 1)
        model.setParam('MIPGap', 0)
        model.Params.Presolve = 1

        return model

    @staticmethod
    def get_cutting_plane_model(title):
        cutting_plane_model: Model = Model(title)

        if not xtra_verbose:
            cutting_plane_model.Params.OutputFlag = 0

        cutting_plane_model.Params.PoolSearchMode = 1
        cutting_plane_model.setParam('GURO_PAR_MINBPFORBID', 1)
        cutting_plane_model.ModelSense = -1
        cutting_plane_model.Params.PoolSolutions = 200
        cutting_plane_model.Params.Presolve = 1


        return cutting_plane_model

    def solve(self):
        model = self.master_problem_model
        
        #made it so you can pick useing a callback or the old way
        if self.callback:
            def find_cluster(m, where):
                if where == GRB.Callback.MIPSOL or where == GRB.Callback.MIPNODE:
                    # Initialise model and get params
                    cutting_plane_model: Model = Model('Cutting plane problem')
                    cutting_plane_model.Params.PoolSearchMode = 1
                    cutting_plane_model.setParam('GURO_PAR_MINBPFORBID', 1)
                    cutting_plane_model.ModelSense = -1
                    cutting_plane_model.Params.PoolSolutions = 200
                    cutting_plane_model.Params.OutputFlag = 1
    
                    if where == GRB.Callback.MIPSOL:
                        solution_set = {(W, u): yeet for ((W, u), yeet) in zip(I.keys(), model.cbGetSolution(I.values()))}
                    else:
                        solution_set = {(W, u): yeet for ((W, u), yeet) in zip(I.keys(), model.cbGetNodeRel(I.values()))}
    #                sl2 = {(W, u): yeet for ((W, u), yeet) in zip(I.keys(), m.getSolution(I.values())) if yeet > 0.001}
    #                print(sl)
    #                print(sl2)
                    # solution_set = {(W, u): I[W, u].x for (W, u) in I.keys()}
                    variable_range = range(dataset.num_variables)
    
                    # VARIABLES
                    J = {(W, u): cutting_plane_model.addVar(
                        vtype=GRB.BINARY) for (W, u) in solution_set.keys() if (solution_set[W, u] > 0.01)}
                    K = {u: cutting_plane_model.addVar(
                        vtype=GRB.BINARY) for u in variable_range}
    
                    #if solver_approach == 'branching':
                    ksum = cutting_plane_model.addVar()
                    cutting_plane_model.addConstr(ksum == quicksum(K.values()))
                    ksum.BranchPriority = 100
    
                    for u in variable_range:
                        K[u].BranchPriority = 10
    
                    # OBJECTIVE
                    cutting_plane_model.setObjective(
                        quicksum(solution_set[W, u] * J[W, u] for (W, u) in J.keys()) - quicksum(
                            K[u] for u in variable_range),
                        GRB.MINIMIZE)
    
                    # CONSTRAINTS
                    # Objective value must be strictly less than 1
                    cutting_plane_model.addLConstr(
                        quicksum(solution_set[W, u] * J[W, u] for (W, u) in J.keys()) - quicksum(
                            K[u] for u in variable_range) >= -0.98)
    
                    # These constraints come from (8) in the paper
                    acyclicity_constraints = {(W, u): cutting_plane_model.addLConstr(
                        (1 - J[W, u]) + quicksum(K[x] for x in W) >= 1) for (W, u) in J.keys()}
    
                    acyclicity_constraints2 = {(W, u): cutting_plane_model.addLConstr(
                        (1 - J[W, u]) + K[u] >= 1) for (W, u) in J.keys()}
    
                    cluster_size_constraint = cutting_plane_model.addLConstr(
                        quicksum(K[u] for u in variable_range) >= 2)
    
                    # OPTIMISE
                    def cutoff(cmodel,where):
                        if where == GRB.Callback.MIP:
                            if cutting_plane_model.cbGet(GRB.Callback.RUNTIME) > 120:
                                cutting_plane_model.terminate()
        
                    cutting_plane_model.optimize(cutoff)
    
                    nsols = cutting_plane_model.Solcount
                    cluster = set()
                    for i in range(1):
                        cutting_plane_model.Params.SolutionNumber = i
                        new_cluster = tuple([u for u in variable_range if K[u].x > 0.01])
                        cluster.add(new_cluster)
    
                    if cutting_plane_model.status == GRB.Status.INFEASIBLE:
                        print('yeeeeet')
                        solution_set = [(W,u) for (W,u) in solution_set.keys() if solution_set[W,u] > 0.001]
                        print(solution_set)
                        print('cluster', cluster)
                        
                     
    
                    for x in cluster:
                        model.cbLazy(
                            quicksum(I[W, u] for u in x for W in parent_sets if intersection_size(W, x) < 1) >= 1)
    
                        model.cbLazy(
                            quicksum(I[W, u] for u in x for W in parent_sets if intersection_size(W, x) < 2) >= 2)
                        
        else:
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


        # Data
        variables = range(self.dataset.num_variables)
        #model.Params.Presolve = 0
        model.setParam('LazyConstraints', 1)
        model.setParam('MIPGap', 0)

        # Initlialise and setup model params

        # Linear variables because we really only care about the linear relaxation
        if genourway:
            added = optimal_extend_path(data.num_variable,{},data)
            #TODO the rest of this eeeee
            #need to setp up I in a bit of a different way
        else:
            I = {(W, u): model.addVar(vtype=GRB.BINARY)
    #        I = {(W, u): model.addVar(ub = 1)
                 for W in self.parent_sets
                 for u in variables}

        model.setObjective(
            quicksum(self.scores[W, u] * I[W, u] for (W,u) in I.keys()), GRB.MAXIMIZE)

        x = tuple(variables)
        model.addLConstr(
            quicksum(I[W, u] for u in x for W in self.parent_sets if intersection_size(W, x) < 1) >= 1)

        model.addLConstr(
            quicksum(I[W, u] for u in x for W in self.parent_sets if intersection_size(W, x) < 2) >= 2)
            

        # Only one parent set
        self.master_convexity_constraints = \
            {u: model.addConstr(quicksum(I[W, u] for W in self.parent_sets) == 1) for u in variables}

        master_start = timer()
        while True:
            # Initialise results from last iteration
            try:
                last_graph = set([(u, W) for (W, u) in I.keys() if I[W, u].x > 0.01])
            except AttributeError:
                last_graph = set()

            # optimise again with cutting constraints from previous iteration
            # model.reset()
            if self.callback:
                model.optimize(find_cluster)
            else:
                model.reset()
                model.optimize()

            # if new constraints from previous iteration render model infeasible:
            if model.status == GRB.Status.INFEASIBLE:
                print("Time taken: {}".format(round(timer() - master_start), 2))
                print("Model infeasible")
                return

            # If no changes to solution after optimisation, then complete
            diff = set([(u, W) for (W, u) in I.keys() if I[W, u].x > 0.001]).difference(last_graph)
            if not diff and not genterway:
                if gentheirway:
                    result = {(W, u): I[W, u].x for (W, u) in I.keys()}
                    added = []
                    for x in results:
                        added.append(extend_path(x,variables,score,data))
                    #TODO stuff hereerererrerererererere
                    #need to add the new var to problem and const
                    if not any(added):
                        #exit somehow here
                    else:
                        #add all vars and const
                        
                else:
                     result = {(W, u): I[W, u].x for (W, u) in I.keys()}
                     result = [(W, u) for (W,u) in result.keys() if result[W, u] > 0.01]
    
                     # self.print_parent_visualisation(result)
                     print("Objective value: {}".format(model.objVal))
                     print("Time taken: {}".format(round(timer()-master_start), 2))
    
                     print_parent_visualisation(result)
    
                     return model

            if not self.callback:
                result = {(W, u): I[W, u].x for (W, u) in I.keys()}
                new_cluster = self.find_cluster(self, variables, result)




                 if new_cluster:
                     self.num_clusters += 1
                     for x in new_cluster:
                         self.clusters.append(x)
                         self.cluster_constr_k1[x] = model.addLConstr(
                             quicksum(I[W, u] for u in x for W in self.parent_sets if self.intersection_size(W, x) < 1),
                             GRB.GREATER_EQUAL, 1)
                
                         self.cluster_constr_k2[x] = model.addLConstr(
                             quicksum(I[W, u] for u in x for W in self.parent_sets if self.intersection_size(W, x) < 2),
                             GRB.GREATER_EQUAL, 2)

    # Helper functions -------------------------------------------------------------------------------------------------
    @staticmethod
    def intersects(W, cluster) -> int:
        return 1 if len([v for v in cluster if v in W]) == 0 else 0

    @staticmethod
    def print_parent_visualisation(res):
        print('Resulting network:')
        for parent_child_set in res:
            parents = parent_child_set[0]
            child = parent_child_set[1]
            print(child, '<-', *parents)
            
def optimal_extend_path(variables,score,data):
    variables = set(variables)
    i = 0
    added = {}
    for x in variables:
        added[x] = set()
    for i in range(5):
        for x in variables:
            test = None
            print("var")
            print(x)
            parents = added[x]
            if len(parents) >= i:
                for y in variables.difference(parents):
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
    return added

def extend_path(nodeset,varset,score,data):
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
            if distence(x,parents,,data,score) <= distence(x,tuple(varset.difference(x)),data,score):
                this = set(parents)
                this = this.union(set(x))
                parents = tuple(sorted(tuple(this)))
                expend = True
                added.add((parents,nodeset[1]))
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


def intersection_size(W, cluster) -> int:
    return len([v for v in cluster if v in W])


def main(data_dir, approach, parent_limit):
    dataset = parse_dataset(data_dir)

    solver = Solver(dataset, parent_limit, approach)
    model = solver.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of exact bayesian network construction via integer programming")
    parser.add_argument("-d", "--datadir", dest="datadir",
                        help="Directory path containing data",
                        metavar="FILE", default='data/mildew_100.data')
    parser.add_argument("-p", "--parentlimit", dest="parentlimit",
                        help="limit to parent set size",
                        metavar="INT", default=2)
    parser.add_argument("-a", "--approach", dest="approach",
                        help="approach to take",
                        metavar="STR", default='branching')
    parser.add_argument("-v", "--verbose", dest='verbose',
                        help="modify output verbosity",
                        action="store_true")
    parser.add_argument("-vv", "--xtra_verbose", dest='xtra_verbose',
                        help="modify output verbosity",
                        action="store_true")
    parser.add_argument("-vvv", "--verbosest", dest='verbosest',
                        help="modify output verbosity",
                        action="store_true")
    args = parser.parse_args()

    # Print statements
    verbose = args.verbose
    xtra_verbose = args.verbose
    verbosest = args.verbosest

    if xtra_verbose:
        verbose = True

    print('Verbose: {}'.format(verbose))
    main(args.datadir, args.approach, int(args.parentlimit))
