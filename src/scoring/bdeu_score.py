from math import lgamma, log

import itertools

import numpy as np

from .caching import are_scores_cached, load_cached_scores, cache_scores

def bdeu_scores(dataset, variables, parent_sets):
    num_parents = max(len(s) for s in parent_sets)
#    if are_scores_cached(dataset.dataset_name, num_parents):
#        return load_cached_scores(dataset.dataset_name, num_parents, variables, parent_sets)

    score_dict = {}
    for variable in variables:        
        for parent in parent_sets:
            if parent:
                combo= []
                for i in range(1,len(parent)+1):
                    for j in itertools.combinations(parent,i):
                        combo.append(j)
            else:
                combo = []
            score = 0
            for x in combo:
                conditional_sample_size = sum([1 if sum([dataset.variables[y][i] for y in x]) == len(x) else 0 for i in range(len(dataset.variables[variable]))])
                var_cardinality = 2
                score += lgamma(var_cardinality) - lgamma(conditional_sample_size + var_cardinality)
                for state in [0,1]:
                    if sum([1 if sum([dataset.variables[y][i] for y in x]) == state*len(x) else 0 for i in range(len(dataset.variables[variable]))]) > 0:
                        score += lgamma(sum([1 if sum([dataset.variables[y][i] for y in x]) == state*len(x) else 0 for i in range(len(dataset.variables[variable]))]) + 1)
            else:
                conditional_sample_size = 1
                var_cardinality = 2
                score += lgamma(var_cardinality) - lgamma(conditional_sample_size + var_cardinality)
                for state in [0,1]:
                    score += lgamma(dataset.variables[variable].count(state) + 1)
                
            score_dict[parent,variable] = score
        print(variable)
    
    #cache_scores(dataset.dataset_name, score_dict, num_parents)

    return score_dict