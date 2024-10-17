import numpy as np
import random
from typing import List, Tuple, Optional

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,      # Population size
        generations: int,   # Number of generations for the algorithm
        mutation_rate: float,  # Gene mutation rate
        crossover_rate: float,  # Gene crossover rate
        tournament_size: int,  # Tournament size for selection
        elitism: bool,         # Whether to apply elitism strategy
        random_seed: Optional[int],  # Random seed for reproducibility
    ):
        # Students need to set the algorithm parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        """
        Initialize the population and generate random individuals, ensuring that every student is assigned at least one task.
        :param M: Number of students
        :param N: Number of tasks
        :return: Initialized population
        """
        # TODO: Initialize individuals based on the number of students M and number of tasks N
        pop = []
        for i in range(self.pop_size):
            # Generate a random assignment of tasks to students
            individual = [random.randint(0, M - 1) for _ in range(N)]  # Ensure student index is in range(0, M)
            pop.append(individual)
        return pop


    def _fitness(self, individual: List[int], student_times: np.ndarray) -> float:
        """
        Fitness function: calculate the fitness value of an individual.
        :param individual: Individual
        :param student_times: Time required for each student to complete each task
        :return: Fitness value
        """
        # TODO: Design a fitness function to compute the fitness value of the allocation plan
        total_time = 0
        for task, student in enumerate(individual):
            task_time = student_times[student][task]
            if task_time == 0:
                # Apply a large penalty for invalid assignments (i.e., tasks that a student cannot perform)
                return float('inf')  # Assign an infinite cost if a student cannot perform the task
            total_time += task_time
            total_time = int(total_time)
        return total_time


    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Use tournament selection to choose parents for crossover.
        :param population: Current population
        :param fitness_scores: Fitness scores for each individual
        :return: Selected parent
        """
        # TODO: Use tournament selection to choose parents based on fitness scores
        tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
        tournament.sort(key=lambda x: x[1])  # Sort by fitness score (lower is better)
        return tournament[0][0]  # Return the individual with the best fitness score

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        """
        Crossover: generate two offspring from two parents.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :param M: Number of students
        :return: Generated offspring
        """
        # TODO: Complete the crossover operation to generate two offspring
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 2)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            offspring1, offspring2 = parent1[:], parent2[:]
        return offspring1, offspring2

    def _mutate(self, individual: List[int], M: int) -> List[int]:
        """
        Mutation operation: randomly change some genes (task assignments) of the individual.
        :param individual: Individual
        :param M: Number of students
        :return: Mutated individual
        """
        # TODO: Implement the mutation operation to randomly modify genes
        for task in range(len(individual)):
            if random.random() < self.mutation_rate:
            # Assign task to a random student
                individual[task] = random.randint(0, M - 1)
        return individual

    def __call__(self, M: int, N: int, student_times: np.ndarray) -> Tuple[List[int], int]:
        """
        Execute the genetic algorithm and return the optimal solution (allocation plan) and its total time cost.
        :param M: Number of students
        :param N: Number of tasks
        :param student_times: Time required for each student to complete each task
        :return: Optimal allocation plan and total time cost
        """
        # TODO: Complete the genetic algorithm process, including initialization, selection, crossover, mutation, and elitism strategy
        population = self._init_population(M, N)
        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = [self._fitness(ind, student_times) for ind in population]

            # Find the best individual in the current population
            if min(fitness_scores) < best_fitness:
                best_fitness = min(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]

            # Selection and reproduction
            new_population = []
            if self.elitism:
                # Keep the best individual (elitism)
                new_population.append(best_individual)

            while len(new_population) < self.pop_size:
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                offspring1, offspring2 = self._crossover(parent1, parent2, M)
                new_population.append(self._mutate(offspring1, M))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate(offspring2, M))

            population = new_population

        return best_individual, best_fitness

if __name__ == "__main__":
    def write_output_to_file(problem_num: int, total_time: float, filename: str = "results.txt") -> None:
        """
        Write results to a file and check if the format is correct.
        """
        print(f"Problem {problem_num}: Total time = {total_time}")

        if not isinstance(total_time, int):
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")
        
        with open(filename, 'a') as file:
            file.write(f"Total time = {total_time}\n")

    # TODO: Define multiple test problems based on the examples and solve them using the genetic algorithm
    # Example problem 1 (define multiple problems based on the given example format)
    # M, N = 2, 3
    # student_times = [[3, 8, 6],
    #                  [5, 2, 7]]

    M1, N1 = 2, 3
    cost1 = [[3,2,4],
             [4,3,2]]
    
    M2, N2 = 4, 4
    cost2 = [[5,6,7,4],
             [4,5,6,3],
             [6,4,5,2],
             [3,2,4,5]]
    
    M3, N3 = 8, 9
    cost3 = [[90, 100, 60, 5, 50, 1, 100, 80, 70],
            [100, 5, 90, 100, 50, 70, 60, 90, 100],
            [50, 1, 100, 70, 90, 60, 80, 100, 4],
            [60, 100, 1, 80, 70, 90, 100, 50, 100],
            [70, 90, 50, 100, 100, 4, 1, 60, 80],
            [100, 60, 100, 90, 80, 5, 70, 100, 50],
            [100, 4, 80, 100, 90, 70, 50, 1, 60],
            [1, 90, 100, 50, 60, 80, 100, 70, 5]]
    
    M4, N4 = 3, 3
    cost4 = [[2,5,6],
             [4,3,5],
             [5,6,2]]
    
    M5, N5 = 4, 4
    cost5 = [[4,5,1,6],
             [9,1,2,6],
             [6,9,3,5],
             [2,4,5,2]]
    
    M6, N6 = 4, 4
    cost6 = [[7,6,4,5],
             [6,4,3,8],
             [8,3,7,6],
             [2,9,8,7]]
    
    M7, N7 = 4, 4
    cost7 = [[16,29.2,34.8,36],[24,12.5,26.1,28],[32,25,8.7,24],[28,33.3,30.4,12]]
    cost7 = [[x * 0.25 for x in sublist] for sublist in cost7]


    
    M8, N8 = 5, 5
    cost8 = [[8,18,30,21,27],
             [24,6,30,21,27],
             [24,18,10,21,27],
             [24,18,30,7,27],
             [24,18,30,21,9]]
    
    M9, N9 = 5, 5
    cost9 = [[10,10,0,0,0],
             [12,0,0,12,12],
             [0,15,15,0,0],
             [11,0,11,0,0],
             [0,14,0,14,14]]
    
    M10, N10 = 9, 10
    cost10 = [[1, 90, 100, 50, 70, 20, 100, 60, 80, 90],
        [100, 10, 1, 100, 60, 80, 70, 100, 50, 90],
        [90, 50, 70, 1, 100, 100, 60, 90, 80, 100],
        [70, 100, 90, 5, 10, 60, 100, 80, 90, 50],
        [50, 100, 100, 90, 20, 4, 80, 70, 60, 100],
        [100, 5, 80, 70, 90, 100, 4, 50, 1, 60],
        [90, 60, 50, 4, 100, 90, 100, 5, 10, 80],
        [100, 70, 90, 100, 4, 60, 1, 90, 100, 5],
        [80, 100, 5, 60, 50, 90, 70, 100, 4, 1]]

    problems = [(M1, N1, np.array(cost1)),
                (M2, N2, np.array(cost2)),
                (M3, N3, np.array(cost3)),
                (M4, N4, np.array(cost4)),
                (M5, N5, np.array(cost5)),
                (M6, N6, np.array(cost6)),
                (M7, N7, np.array(cost7)),
                (M8, N8, np.array(cost8)),
                (M9, N9, np.array(cost9)),
                (M10, N10, np.array(cost10))]

    # Example for GA execution:
    # TODO: Please set the parameters for the genetic algorithm
    ga = GeneticAlgorithm(
    pop_size=100,
    generations=1000,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=3,
    elitism=True,
    random_seed=42  # 确保这是一个整数
    )

    # Solve each problem and immediately write the results to the file
    for i, (M, N, student_times) in enumerate(problems, 1):
        best_allocation, total_time = ga(M=M, N=N, student_times=student_times)
        write_output_to_file(i, total_time)

    print("Results have been written to results.txt")
