from gurobipy import *
import argparse
from scoring.scoring import bdeu_scores, bdeu_scores_sig, score_parents
from data import Dataset, parse_dataset
from timeit import default_timer as timer
from math_utils import binomial_coefficient, factorial, get_subsets_of_size, parse_number


class Solver:
    """
    Problem solver class for the MIP.
    """
    def __init__(self, dataset, parent_set_limit, solver_approach='branching'):
        # General data for the problem 
        self.dataset = dataset
        self.parent_lim = parent_set_limit
        self.parent_sets = [s for s in get_subsets_of_size(dataset.num_variables, self.parent_lim)]
        self.scores = bdeu_scores(self.dataset, self.parent_sets)
        
        # Problem configs 
        self.solver_approach = solver_approach
        
        # Models
        self.master_problem_model = self.get_master_model('Bayesian Network Learning')
        self.cutting_plane_models = {}

        # Master problem constraints
        self.master_convexity_constraints = {}

        # Cutting plane constraints
        self.clusters = []
        self.num_clusters = 0
        self.cluster_constr_k1 = {}
        self.cluster_constr_k2 = {}

    @staticmethod
    def get_master_model(title):
        model = Model(title)
        model.ModelSense = -1               # maximise objective
        model.Params.PreCrush = 1           # since (always) adding cuts
        model.Params.CutPasses = 100000     # want to allow many cuts
        model.Params.GomoryPasses = 100000  # want to allow many cuts
        model.Params.MIPFocus = 2           # focus on proving optimality
        if not verbose:
            model.Params.OutputFlag = 0         #
        return model

    @staticmethod
    def get_cutting_plane_model(title):
        cutting_plane_model: Model = Model('Cutting Plane')
        cutting_plane_model.Params.OutputFlag = 0
        cutting_plane_model.setParam('GURO_PAR_MINBPFORBID', 1)
        cutting_plane_model.ModelSense = -1
        cutting_plane_model.Params.PoolSolutions = 200
        cutting_plane_model.Params.PoolSearchMode = 1
        return cutting_plane_model

    def solve(self):
        # Data
        variables = range(self.dataset.num_variables)

        # Initlialise and setup model params
        model = self.master_problem_model

        # Linear variables because we really only care about the linear relaxation
        I = {(W, u): model.addVar(vtype=GRB.BINARY)
             for W in self.parent_sets
             for u in variables}

        model.setObjective(
            quicksum(self.scores[W, u] * I[W, u] for (W,u) in I.keys()), GRB.MAXIMIZE)

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
            model.reset()
            model.optimize()

            # if new constraints from previous iteration render model infeasible:
            if model.status == GRB.Status.INFEASIBLE:
                print("Time taken: {}".format(round(timer() - master_start), 2))
                print("Model infeasible")
                return

            # If no changes to solution after optimisation, then complete
            diff = set([(u, W) for (W, u) in I.keys() if I[W, u].x > 0.001]).difference(last_graph)
            if not diff:
                result = {(W, u): I[W, u].x for (W, u) in I.keys()}
                result = [(W, u) for (W,u) in result.keys() if result[W, u] > 0.01]

                self.print_parent_visualisation(result)
                print("Objective value: {}".format(model.objVal))
                print("Time taken: {}".format(round(timer()-master_start), 2))

                return model

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

    @staticmethod
    def find_cluster(self, variable_range, solution_set):

        # Initialise model and get params
        cutting_plane_model: Model = self.get_cutting_plane_model('Cutting plane problem')

        # VARIABLES
        J = {(W, u): cutting_plane_model.addVar(
            vtype=GRB.BINARY) for (W, u) in solution_set.keys() if (solution_set[W,u] > 0.01)}
        K = {u: cutting_plane_model.addVar(
            vtype=GRB.BINARY) for u in variable_range}

        if self.solver_approach == 'branching':
            ksum = cutting_plane_model.addVar()
            cutting_plane_model.addConstr(ksum == quicksum(K.values()))
            ksum.BranchPriority = 100

        for u in variable_range:
            K[u].BranchPriority = 10

        # OBJECTIVE
        cutting_plane_model.setObjective(
            quicksum(solution_set[W,u]*J[W, u] for (W, u) in J.keys()) - quicksum(K[u] for u in variable_range),
            GRB.MINIMIZE)

        # CONSTRAINTS
        # Objective value must be strictly less than 1
        cutting_plane_model.addConstr(
            quicksum(solution_set[W, u] * J[W, u] for (W, u) in J.keys()) - quicksum(K[u] for u in variable_range) >= -0.98)

        # These constraints come from (8) in the paper
        acyclicity_constraints = {(W, u): cutting_plane_model.addLConstr(
                (1-J[W, u]) + quicksum(K[x] for x in W) >= 1) for (W,  u) in J.keys()}

        acyclicity_constraints2 = {(W, u): cutting_plane_model.addLConstr(
                (1-J[W, u]) + K[u] >= 1) for (W,u) in J.keys()}

        cluster_size_constraint = cutting_plane_model.addLConstr(
                quicksum(K[u] for u in variable_range) >= 2)

        # OPTIMISE
        cutting_plane_model.optimize()

        nsols = cutting_plane_model.Solcount
        cluster = set()
        for i in range(nsols):
            cutting_plane_model.Params.SolutionNumber = i
            new_cluster = tuple([u for u in variable_range if K[u].x > 0.01 ])
            cluster.add(new_cluster)

        self.cutting_plane_models[self.num_clusters] = cutting_plane_model
        return cluster

    # Helper functions -------------------------------------------------------------------------------------------------
    @staticmethod
    def intersects(W, cluster) -> int:
        return 1 if len([v for v in cluster if v in W]) == 0 else 0

    @staticmethod
    def intersection_size(W, cluster) -> int:
        return len([v for v in cluster if v in W])

    @staticmethod
    def print_parent_visualisation(res):
        print('Resulting network:')
        for parent_child_set in res:
            parents = parent_child_set[0]
            child = parent_child_set[1]
            print(child, '<-', *parents)


def main(data_dir, approach, parent_limit):
    dataset = parse_dataset(data_dir)

    solver = Solver(dataset, parent_limit, approach)
    model = solver.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of exact bayesian network construction via integer programming")
    parser.add_argument("-d", "--datadir", dest="datadir",
                        help="Directory path containing data",
                        metavar="FILE", default='data/Mildew_100.data')
    parser.add_argument("-p", "--parentlimit", dest="parentlimit",
                        help="limit to parent set size",
                        metavar="INT", default=2)
    parser.add_argument("-a", "--approach", dest="approach",
                        help="approach to take",
                        metavar="STR", default='branching')
    parser.add_argument("-v", "--verbose", dest='verbose',
                        help="modify output verbosity",
                        action="store_true")

    args = parser.parse_args()
    verbose = args.verbose
    print('Verbose: {}'.format(verbose))
    main(args.datadir, args.approach, int(args.parentlimit))
