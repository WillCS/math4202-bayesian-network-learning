import os
from math_utils import parse_number

DATA_DIR_NAME = 'data'


class Dataset():
    def __init__(self, variables, variable_sizes):
        #not this is a hyperparameter we can mess around with
        self.sample_size = 10
        
        self.variables = variables
        self.num_variables = len(variables)
        self.num_observations = len(variables[0])
        self.variable_sizes = variable_sizes

    def set_dataset_name(self, name: str) -> None:
        self.dataset_name = name

    def p_of(self, variable: int, value) -> float:
        return self.variables[variable].count(value) / self.num_observations

    def conditional_p_of(self, variable: int, value, evidence: [int], e_values: []) -> float:
        return [
            self.variables[variable][observation] for observation in range(self.num_observations) 
                    if all(self.variables[e][observation] == e_value[s] for e in evidence)
        ].count(value) / self.num_observations


def parse_dataset(file_name) -> Dataset:
    try:
        dataset_path = os.path.abspath(f'{file_name}')

        with open(dataset_path, 'r') as file:
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

            dataset = Dataset(variables, variable_sizes)
            dataset.set_dataset_name(file_name)
            return dataset
    except FileNotFoundError:
        print(f'File not found: {file_name}')
