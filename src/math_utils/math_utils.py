from functools import reduce
from itertools import combinations, chain
from collections import Iterable

def parse_number(number_string: str):
    try:
        return int(number_string)
    except TypeError:
        return float(number_string)

def factorial(n: int) -> int:
    if n < 0:
        return -1
    elif n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def binomial_coefficient(n: int, k: int) -> int:
    k_range = range(1, k + 1)
    return reduce((lambda x, y: x * y), 
            map((lambda i: (n + 1 - i) / i), k_range))

def get_subsets_of_size(all_elements: Iterable, size: int) -> Iterable:
    all_elements = range(all_elements)
    x = combinations(all_elements, 0)
    for i in range(1,size+1):
        x = chain(x, combinations(all_elements, i))
    return x