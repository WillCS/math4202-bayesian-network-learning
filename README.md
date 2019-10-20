# MATH4202 Bayesian Network Learning with Integer Programming

Project implementation for MATH4202 - _Advanced Topics in Operations Research_ based on 
Cussens' [Bayesian network learning with cutting planes](https://arxiv.org/abs/1202.3713)
and Cussens et al.'s 
[Bayesian Network Structure Learning with Integer Programming: Polytopes, Facets and Complexity](https://www.ijcai.org/proceedings/2017/708).
 
 ## tldr
 We're implementing Bayesian network learning using integer programming; first by implementing it
 as described in the papers above, and then attempting to improve upon it.
 
 ## requirements
 * Python 3.7.3 or newer
 * Gurobi 8.0+ for Python
 
 ## data
 ```
 python get_data.py [filters...]
 ```
 `get_data.py` will download all of the data used in the original papers into a new directory, `gobnilp`. Optional filters
 can be provided, in which case only files containing at lease one of the filter strings will be downloaded.
 
 Alternatively, data is provided in the repository, within the `data` directory 

## usage 
The program may solve the problem with the given data through different implementations via commandline. 
To run the original implementation, run from the root directory of the project: 

```
python3.7 src/our_formulation.py -d data/asia_1000.data -v
```

Here, `-v` enables `verbose`, which prints solver information to the console. The program may be run without the flag
There are three levels of verbosity in total (`-v`, `-vv`, `-vvv`, with each getting more and more verbose). 

The directory for the data to learn the BN from is indicated by the `-d` flag followed by the path of the directory. 

#### Run with branch variables 
To run with our implemented branch variables, simply add a `-b`: 
```
python3.7 src/our_formulation.py -d data/asia_1000.data -b -v 
```
#### Run with callback 
Or to run with callback and branch variables, add `-c`: 
```
python3.7 src/our_formulation.py -d data/asia_1000.data -b -c -v 
``` 

#### Run with optimised path extension
To run with optimised path extension, either before or after solving the master problem respectively, run: 
```
python3.7 src/our_formulation.py -d data/asia_1000.data -b -c -o before 
``` 

```
python3.7 src/our_formulation.py -d data/asia_1000.data -b -c -o after 
``` 

Below is the usage for the program which can be seen using the `-h` flag with running the script:  
```
python3.7 src/our_formulation.py -d data/asia_100.data -h               ✔  ⚙  09:00:24
usage: our_formulation.py [-h] [-d FILE] [-p INT] [-b] [-o STR] [-c] [-v]
                          [-vv] [-vvv]

Implementation of exact bayesian network construction via integer programming

optional arguments:
  -h, --help            show this help message and exit
  -d FILE, --datadir FILE
                        Directory path containing data
  -p INT, --parentlimit INT
                        limit to parent set size
  -b, --branchvars      optimise via configuring branching variables
  -o STR, --optimalpath STR
                        how to extend path = {before, after,none}
  -c, --callback        modify output verbosity
  -v, --verbose         modify output verbosity
  -vv, --extra_verbose  modify output verbosity
  -vvv, --verbosest     modify output verbosity

```
 
 If you prefer to not use commandline, then there is a main function to run the problem from: 
```
main(data_dir, parent_limit, branchvars, optimalpath, callback):
```

where: 
```    """
def main(data_dir, parent_limit, branchvars, optimalpath, callback):
    Driver function for the solver.
    Args:
        data_dir: string path for the data file
        parent_limit: int: maximum number of parents in each parent set in the network
        branchvars: bool: whether to use branch variables in the solve or not
        optimalpath: {None, 'before', 'after'} whether to use branch variables before solving the problem,
                        or optimising the problem further after solving the master problem, or not at all.
        callback: bool: whether to use callbacks to solve the problem

    Returns:
        the model of the master problem solved (or infeasible model if problem not feasible).
    """
```