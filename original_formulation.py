from gurobipy import *

def parse_number(number_string: str):
    try:
        return int(number_string)
    except TypeError:
        return float(number_string)


class Dataset():
    def __init__(self, variables, variable_sizes):
        self.variables = variables
        self.num_variables = len(variables)
        self.num_observations = len(variables[0])
        self.variable_sizes = variable_sizes

    def p_of(self, variable: int, value) -> float:
        return self.variables[variable].count(value) / self.num_observations

    def posterior_p_of(self, variable: int, value, evidence: int, e_value) -> float:
        return [
            self.variables[variable] for e in self.variables[evidence] if e == e_value
        ].count(value) / self.num_observations

def parse_dataset(file_name: str) -> Dataset:
    try:
        with open(file_name, 'r') as file:
            line_num: int = 0

            num_variables: int = 0
            variables: [[]] = None

            variable_sizes: [[]] = None

            for line in file:
                line_num += 1

                if line_num > 3:
                    observation = [parse_number(num) for num in line.split()]
                    observation_index = line_num - 4

                    for v in range(num_variables):
                        variables[v][observation_index] = observation[v]
                elif line_num == 3:
                    num_observations = int(line)
                    variables = [[0 for n in range(num_observations)] for v in range(num_variables)]
                elif line_num == 2:
                    variable_sizes = [parse_number(num) for num in line.split()]
                else:
                    num_variables = int(line)

            return Dataset(variables, variable_sizes)
    except FileNotFoundError:
        print(f'File not found: {file_name}')

dataset = parse_dataset('data/mildew_10000.data')

for i in range(3):
    print(dataset.p_of(0, i))