# math4202-bayesian-network-learning

Project implementation for MATH4202 - _Advanced Topics in Operations Research_ based on 
Cussens' [Bayesian network learning with cutting planes](https://arxiv.org/abs/1202.3713)
and Cussens et al.'s 
[Bayesian Network Structure Learning with Integer Programming: Polytopes, Facets and Complexity](https://www.ijcai.org/proceedings/2017/708).
 
 ## tldr
 We're implementing Bayesian network learning using integer programming; first by implementing it
 as described in the papers above, and then attempting to improve upon it.
 
 ## requirements
 * Python 3.7.3 or newer
 * Gurobi for Python
 
 ## data
 ```
 python get_data.py [filters...]
 ```
 `get_data.py` will download all of the data used in the original papers into a new directory, `gobnilp`. Optional filters
 can be provided, in which case only files containing at lease one of the filter strings will be downloaded.
 
 ## improvements?
 * Delayed Column Generation
 * Some wierd branching variable magic idk
