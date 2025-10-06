# de_optimizer.py (Updated version)
import numpy as np
import matplotlib.pyplot as plt

class DifferentialEvolution:
    """
    Differential Evolution optimizer for drilling parameter optimization
    """
    
    def __init__(self, pop_size=20, F=0.5, CR=0.7, max_iter=100):
        """
        Initialize DE optimizer
        
        Parameters:
        -----------
        pop_size : int, population size
        F : float, mutation factor (0 < F <= 2)
        CR : float, crossover probability (0 <= CR <= 1)
        max_iter : int, maximum iterations
        """
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
    
    def objective_function(self, params, rop_model, torque_model, fixed_params):
        """
        Multi-objective function: maximize ROP while keeping torque in range
        
        Parameters:
        -----------
        params : array [WOB, RPM]
        rop_model : trained XGBoost model for ROP prediction
        torque_model : trained XGBoost model for torque prediction
        fixed_params : array [SPP, Q, Depth, Hook_Load]
        
        Returns:
        --------
        fitness : float (negative ROP + torque penalty)
        """
        # Combine optimized and fixed parameters
        full_params = np.concatenate([params, fixed_params])
        
        # Predict ROP and Torque
        rop = rop_model.predict(full_params.reshape(1, -1))[0]
        torque = torque_model.predict(full_params.reshape(1, -1))[0]
        
        # Convert torque to Lb.Ft
        torque_lbft = torque * 0.737562
        
        # Torque constraints (13,000 - 19,000 Lb.Ft)
        torque_min = 13000
        torque_max = 19000
        
        # Calculate torque penalty
        if torque_lbft < torque_min:
            torque_penalty = (torque_min - torque_lbft) ** 2
        elif torque_lbft > torque_max:
            torque_penalty = (torque_lbft - torque_max) ** 2
        else:
            torque_penalty = 0
        
        # Fitness: minimize negative ROP + torque penalty
        fitness = -rop + 0.01 * torque_penalty
        
        return fitness
    
    def initialize_population(self, bounds):
        """Initialize population within bounds"""
        population = []
        for _ in range(self.pop_size):
            individual = [np.random.uniform(low, high) for low, high in bounds]
            population.append(individual)
        return np.array(population)
    
    def mutate(self, population, current_idx):
        """
        Mutation: v_i = x_r1 + F * (x_r2 - x_r3)
        """
        indices = list(range(len(population)))
        indices.remove(current_idx)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        
        donor = population[r1] + self.F * (population[r2] - population[r3])
        return donor
    
    def crossover(self, target, donor):
        """
        Binomial crossover
        """
        trial = target.copy()
        j_rand = np.random.randint(len(target))
        
        for j in range(len(target)):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = donor[j]
        
        return trial
    
    def selection(self, target, trial, fitness_target, fitness_trial):
        """
        Selection: greedy selection
        """
        if fitness_trial <= fitness_target:
            return trial, fitness_trial
        else:
            return target, fitness_target
    
    def enforce_bounds(self, vector, bounds):
        """Enforce parameter bounds"""
        bounded_vector = vector.copy()
        for i, (low, high) in enumerate(bounds):
            bounded_vector[i] = np.clip(vector[i], low, high)
        return bounded_vector
    
    def optimize(self, bounds, rop_model, torque_model, fixed_params, verbose=True):
        """
        Main optimization loop using XGBoost models
        
        Parameters:
        -----------
        bounds : list of tuples, bounds for [WOB, RPM]
        rop_model : trained XGBoost model for ROP
        torque_model : trained XGBoost model for torque
        fixed_params : array, fixed parameters [SPP, Q, Depth, Hook_Load]
        verbose : bool, print progress
        
        Returns:
        --------
        results : dict with optimization results
        """
        # Initialize population
        population = self.initialize_population(bounds)
        fitness = np.array([
            self.objective_function(ind, rop_model, torque_model, fixed_params) 
            for ind in population
        ])
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        fitness_history = [best_fitness]
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("Starting Drilling Parameter Optimization")
            print(f"{'=' * 70}")
            print(f"Initial Best Fitness: {best_fitness:.4f}\n")
        
        # Evolution loop
        for generation in range(self.max_iter):
            new_population = []
            new_fitness = []
            
            for i in range(self.pop_size):
                # Mutation
                donor = self.mutate(population, i)
                donor = self.enforce_bounds(donor, bounds)
                
                # Crossover
                trial = self.crossover(population[i], donor)
                trial = self.enforce_bounds(trial, bounds)
                
                # Evaluation
                fitness_trial = self.objective_function(
                    trial, rop_model, torque_model, fixed_params
                )
                
                # Selection
                selected, selected_fitness = self.selection(
                    population[i], trial, fitness[i], fitness_trial
                )
                
                new_population.append(selected)
                new_fitness.append(selected_fitness)
                
                # Update best solution
                if selected_fitness < best_fitness:
                    best_solution = selected.copy()
                    best_fitness = selected_fitness
            
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            fitness_history.append(best_fitness)
            
            if verbose and (generation + 1) % 20 == 0:
                print(f"Generation {generation + 1}/{self.max_iter}: Best fitness = {best_fitness:.4f}")
        
        # Calculate final predictions
        full_params = np.concatenate([best_solution, fixed_params])
        final_rop = rop_model.predict(full_params.reshape(1, -1))[0]
        final_torque = torque_model.predict(full_params.reshape(1, -1))[0] * 0.737562
        
        results = {
            'optimal_wob': best_solution[0],
            'optimal_rpm': best_solution[1],
            'predicted_rop': final_rop,
            'predicted_torque': final_torque,
            'fitness_history': fitness_history,
            'final_fitness': best_fitness
        }
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("Optimization Results:")
            print(f"{'=' * 70}")
            print(f"  Optimal WOB:        {results['optimal_wob']:.2f}")
            print(f"  Optimal RPM:        {results['optimal_rpm']:.2f}")
            print(f"  Predicted ROP:      {results['predicted_rop']:.2f} m/hr")
            print(f"  Predicted Torque:   {results['predicted_torque']:.2f} Lb.Ft")
            print(f"{'=' * 70}\n")
        
        return results
    
    def plot_convergence(self, fitness_history):
        """Plot optimization convergence"""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, linewidth=2)
        plt.title('Differential Evolution Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
