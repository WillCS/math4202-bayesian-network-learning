from math import log, lgamma

from .caching import are_scores_cached, load_cached_scores, cache_scores

def bdeu_scores(dataset, variables, parent_sets):
    num_parents = max(len(s) for s in parent_sets)
    if are_scores_cached(dataset.dataset_name, num_parents):
        return load_cached_scores(dataset.dataset_name, num_parents, variables, parent_sets)

    score_dict = {}
    for variable in variables:        
        for parent in parent_sets:
            score = 0
            conditional = 0
            for x in parent:
                conditional += dataset.variable_sizes[x]
            score += (lgamma(dataset.sample_size / len(parent_sets)) - lgamma(conditional + dataset.sample_size / len(parent_sets)))
            
            obs = dataset.variables[variable]
            obs_dict ={}
            for i in range(len(obs)):
                obsis = []
                for y in parent:
                    obsis.append(dataset.variables[y][i])
                if not (tuple(obsis) in obs_dict):
                    obs_dict[tuple(obsis)] = 0
                else:
                    ret_list = obs_dict[tuple(obsis)]
                    ret_list += 1
                    obs_dict[tuple(obsis)] = ret_list
            for x in obs_dict.keys():
                score += (lgamma(obs_dict[x] + dataset.sample_size / (len(parent_sets) * dataset.num_variables)) -
                              lgamma(dataset.sample_size / (len(parent_sets) * dataset.num_variables)))
                
            score_dict[parent,variable] = log(score)
        print(variable)
    
    cache_scores(dataset.dataset_name, score_dict, num_parents)

    return score_dict