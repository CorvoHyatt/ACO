import random
import time
import networkx as nx


class ACO:
    def __init__(
        self,
        grafo,
        num_colors,
        num_ants,
        alpha=1,
        beta=2,
        rho=0.1,
        objective_function=None,
        stopping_criteria_type="iteraciones",
        max_iterations=100,
        max_function_calls=1000,
    ):
        self.grafo = grafo
        self.num_colors = num_colors
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.objective_function = objective_function
        self.stopping_criteria_type = stopping_criteria_type
        self.max_iterations = max_iterations
        self.max_function_calls = max_function_calls

        self.num_nodes = grafo.number_of_nodes()
        self.pheromone = [
            [1.0 for _ in range(num_colors)] for _ in range(self.num_nodes)
        ]

        self.best_coloring = None
        self.best_objective_value = float("inf")
        self.function_call_counter = 0
        self.iteration_counter = 0

    def stopping_criteria(self):
        """Verifica si se cumple el criterio de parada."""
        if self.stopping_criteria_type == "function_calls":
            return self.function_call_counter >= self.max_function_calls
        elif self.stopping_criteria_type == "iteraciones":
            return self.iteration_counter >= self.max_iterations
        else:
            raise ValueError("Tipo de criterio de parada no válido")

    def run(self):
        start_time = time.time()  # Registrar tiempo de inicio
        self._cost_best = []  # Lista para almacenar la mejor solución en cada iteración
        while not self.stopping_criteria():
            self.iteration_counter += 1
            for ant in range(self.num_ants):
                coloring = self.run_ant()
                objective_value = self.objective_function(coloring, self.grafo)
                self.function_call_counter += 1
                if objective_value < self.best_objective_value:
                    self.best_coloring = coloring
                    self.best_objective_value = objective_value

                self._cost_best.append(
                    self.best_objective_value
                )  # Guardar mejor solución en cada iteración

                self.update_pheromone(coloring, objective_value)

        end_time = time.time()  # Registrar tiempo de finalización
        elapsed_time = end_time - start_time
        print(f"Tiempo de ejecución: {elapsed_time:.5f} segundos")

        return self.best_coloring, self.best_objective_value

    def run_ant(self):
        """Función que ejecuta una hormiga."""

        coloring = [-1] * self.num_nodes
        for node in self.grafo.nodes():
            color_probs = self.calculate_color_probabilities(coloring, node)
            coloring[node] = random.choices(
                range(self.num_colors), weights=color_probs
            )[0]

        return coloring

    def calculate_color_probabilities(self, coloring, node):
        """Calcula las probabilidades de elegir cada color para un nodo."""

        color_scores = [0.0] * self.num_colors
        for color in range(self.num_colors):
            neighbors = list(self.grafo.neighbors(node))
            color_scores[color] = (self.pheromone[node][color] ** self.alpha) * (
                1
                / (1 + sum(1 for neighbor in neighbors if coloring[neighbor] == color))
            ) ** self.beta

        total_score = sum(color_scores)
        color_probs = [score / total_score for score in color_scores]
        return color_probs

    def update_pheromone(self, coloring, objective_value):
        """Actualiza la cantidad de feromonas en los caminos."""

        for node in range(self.num_nodes):
            for color in range(self.num_colors):
                self.pheromone[node][color] *= 1 - self.rho
                if coloring[node] == color:
                    self.pheromone[node][color] += 1 / (1 + objective_value)
