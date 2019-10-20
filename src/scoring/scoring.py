import itertools
import random
from math import lgamma, log
from scipy.special import comb

from .caching import are_scores_cached, load_cached_scores, cache_scores

random.seed(13)


def bdeu_scores(dataset, parentsets):
    """
    Calculates bdeu scores for the given dataset and parent sets
    :param dataset: parsed dataset
    :param parentsets: candidate parent sets
    :return: dictionary of scores {((Parent Nodes, Child): Score)}
    """
    num_parents = max(len(s) for s in parentsets)
    if are_scores_cached(dataset.dataset_name, num_parents):
        print('Loading Cached scores at {}'.format(dataset.dataset_name))
        return load_cached_scores(dataset.dataset_name, num_parents, dataset.num_variables, parentsets)
    print('Scores are not cached, calculating scores')

    score_dict = {}

    for variable in range(dataset.num_variables):
        for parent in parentsets:
            if parent:
                combo = []
                for i in range(1, len(parent)+1):
                    for j in itertools.combinations(parent, i):
                        combo.append(j)
            else:
                combo = []
            score = 0
            for x in combo:
                conditional_sample_size = \
                    sum([1 if sum([dataset.variables[y][i] for y in x]) == len(x)*dataset.variables[variable][i]
                         else 0 for i in range(len(dataset.variables[variable]))])

                var_cardinality = dataset.variable_sizes[variable]
                score += lgamma(var_cardinality) - lgamma(conditional_sample_size + var_cardinality)

                for state in range(dataset.variable_sizes[variable]):
                    if sum([1 if sum([dataset.variables[y][i] for y in x]) == state*len(x)
                            else 0 for i in range(len(dataset.variables[variable]))]) > 0:

                        score += \
                            lgamma(sum([1 if sum([dataset.variables[y][i] for y in x]) == state*len(x)
                                        else 0 for i in range(len(dataset.variables[variable]))]) + 1)
            else:
                conditional_sample_size = 1
                var_cardinality = dataset.variable_sizes[variable]
                score += lgamma(var_cardinality) - lgamma(conditional_sample_size + var_cardinality)
                for state in range(dataset.variable_sizes[variable]):
                    score += lgamma(dataset.variables[variable].count(state) + 1)

            score_dict[parent, variable] = score

    cache_scores(dataset.dataset_name, score_dict, num_parents)

    return score_dict


def bdeu_scores_sig(dataset, variable, parent):
    """

    :param dataset: parsed dataset
    :param variable: the variable to calculate the score at
    :param parent: the parent set to calculate the score at
    :return: score of parent set and child family
    """
    if parent:
        if len(parent) > 1:
            size = 0
            combo = itertools.combinations(parent, 1)
            for i in range(2, len(parent)+1):
                size += comb(len(parent), i)
                combo = itertools.chain(combo, itertools.combinations(parent, i))
        else:
            combo = [parent]
    else:
        combo = []
    score = 0
    i = 0
    for x in combo:
        conditional_sample_size = \
            sum([1 if sum([dataset.variables[y][i] for y in x]) == len(x)*dataset.variables[variable][i]
                 else 0 for i in range(len(dataset.variables[variable]))])

        var_cardinality = dataset.variable_sizes[variable]
        score += lgamma(var_cardinality) - lgamma(conditional_sample_size + var_cardinality)

        for state in range(dataset.variable_sizes[variable]):

            if sum([1 if sum([dataset.variables[y][i] for y in x]) == state*len(x)
                    else 0 for i in range(len(dataset.variables[variable]))]) > 0:

                score += lgamma(sum([1 if sum([dataset.variables[y][i] for y in x]) ==
                                          state*len(x)
                                     else 0 for i in range(len(dataset.variables[variable]))]) + 1)
        i += 1
    else:
        conditional_sample_size = 1
        var_cardinality = dataset.variable_sizes[variable]
        score += lgamma(var_cardinality) - lgamma(conditional_sample_size + var_cardinality)
        for state in range(dataset.variable_sizes[variable]):
            score += lgamma(dataset.variables[variable].count(state) + 1)

    return score


def score_parents(parent_sets, variable_set, scoring_type='RANDOM', scoredict=None):
    """
    Score parent-child family sets based on random values or the actual value of the child node.
    :param parent_sets:     candidate parent sets, W
    :param variable_set:    candidate variables, u
    :param scoring_type:    {RANDOM, VALUE} With RANDOM being a random score assigned,
                            and VALUE being the numerical node label
    :param scoredict:        existing score dictionary, can be None
    :return: new updated scores, in scoredict
    """
    if scoredict:
        for W in parent_sets:
            for u in variable_set:
                scoredict[W, u] = random.random()
        return scoredict
    if scoring_type == 'VALUE':
        score = {(W, u): u for W in parent_sets for u in variable_set}
    else:
        score = {(W, u): random.random() for W in parent_sets for u in variable_set}
    return score
