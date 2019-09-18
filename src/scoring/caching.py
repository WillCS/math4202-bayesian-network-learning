import os
import re

CACHE_DIR_NAME = 'cache'

def get_path_to_score_cache(dataset_name: str, parents: int) -> os.path:
    return os.path.abspath(f'{CACHE_DIR_NAME}/{dataset_name}_{parents}p.scores')

def are_scores_cached(dataset_name: str, parents: int) -> bool:
    return os.path.isfile(get_path_to_score_cache(dataset_name, parents))

def cache_scores(dataset_name: str, score_dict: {}, parents: int) -> None:
    if are_scores_cached(dataset_name, parents):
        raise FileExistsError(f'Scores already cached for {dataset_name}')
    lines: [str] = []

    for (W, u), score in score_dict.items():
        new_line = f'{u} ({" ".join(str(w) for w in W)}): {score}\n'
        lines.append(new_line)

    with open(get_path_to_score_cache(dataset_name, parents), 'w') as score_file:
        score_file.writelines(lines)

def load_cached_scores(dataset_name: str, parents: int, variables, parent_sets):
    if are_scores_cached(dataset_name, parents):
        score_dict = {}
        regex: re.Pattern = re.compile(r'(\d+) \((\d+(?: \d+)*)?\): (-?(?:.|\d)+)')

        with open(get_path_to_score_cache(dataset_name, parents), 'r') as score_file:
            for line in score_file:
                results = regex.match(line)

                if results:
                    var, ps, score = results.group(1, 2, 3)

                    if not ps:
                        ps = parent_sets[0]
                    else:
                        ps = tuple([int(p) for p in ps.split(' ')])
                    
                    score_dict[ps, int(var)] = float(score)     

        return score_dict
    else:
        raise FileNotFoundError('Cache Not found.')