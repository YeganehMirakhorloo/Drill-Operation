import numpy as np
import matplotlib.pyplot as plt

class DifferentialEvolution:
    def __init__(self, pop_size=50, F=0.5, CR=0.7, max_iter=100):
        """
        Differential Evolution algorithm as described in the paper
        
        Parameters:
        - pop_size: Population size
        - F: Mutation factor (0 to 2)
        - CR: Crossover probability (0 to 1)
        - max_iter: Maximum iterations
        """
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        
    def objective_function(self, x, ann_model, fixed_params):
        """
        Multi-objective function as described in the paper
        Maximize ROP while keeping torque within limits (13k to 19k Lb.Ft)
        
        x: [WOB, RPM] - optimization variables
        fixed_params: [SPP, Q, Depth, Hook_Load] - fixed drilling parameters
        """
        # Combine optimization variables with fixed parameters
        full_params = np.concatenate([x, fixed_params])
        
        try:
            # Predict ROP and Torque
            predicted_rop = ann_model.predict_rop(full_params)
            predicted_torque = ann_model.predict_torque(full_params)
            
            # Convert torque to Lb.Ft if needed (assuming model predicts in N⋅m)
            # 1 N⋅m = 0.737562 lb⋅ft
            torque_lb_ft = predicted_torque * 0.737562
            
            # Torque constraints (13,000 to 19,000 Lb.Ft as mentioned in paper)
            torque_min = 13000
            torque_max = 19000
            
            # Penalty for torque outside limits
            if torque_lb_ft < torque_min:
                torque_penalty = (torque_min - torque_lb_ft) ** 2
            elif torque_lb_ft > torque_max:
                torque_penalty = (torque_lb_ft - torque_max) ** 2
            else:
                torque_penalty = 0
            
            # Objective: Maximize ROP (minimize negative ROP) with torque penalty
            fitness = -predicted_rop + 0.1 * torque_penalty
            
            return fitness
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e6  # Large penalty for invalid solutions
    
    def initialize_population(self, bounds):
        """
        Initialize population randomly within bounds
        bounds: [(min_wob, max_wob), (min_rpm, max_rpm)]
        """
        pop = np.random.uniform(
            low=[bound[0] for bound in bounds],
            high=[bound[1] for bound in bounds],
            size=(self.pop_size, len(bounds))
        )
        return pop
    
    def mutate(self, population, target_idx):
        """
        Mutation operation: U_{i,j}^{g+1} = X_{k1,i}^{g} + F * (X_{k2,i}^{g} - X_{k3,i}^{g})
        """
        # Select three random indices different from target
        candidates = list(range(len(population)))
        candidates.remove(target_idx)
        k1, k2, k3 = np.random.choice(candidates, 3, replace=False)
        
        # Create donor vector
        donor = population[k1] + self.F * (population[k2] - population[k3])
        
        return donor
    
    def crossover(self, target, donor):
        """
        Crossover operation as described in paper:
        T_{i,j}^{g+1} = { U_{i,j}^{g+1}, if rand_{i,j} ≤ CR or j = I_rand
                         { X_{i,j}^{g},  if rand_{i,j} > CR and j ≠ I_rand
        """
        trial = target.copy()
        
        # Ensure at least one parameter is taken from donor
        I_rand = np.random.randint(0, len(target))
        
        for j in range(len(target)):
            rand_j = np.random.random()
            if rand_j <= self.CR or j == I_rand:
                trial[j] = donor[j]
        
        return trial
    
    def selection(self, target, trial, fitness_target, fitness_trial):
        """
        Selection operation: choose better vector
        X_i^{g+1} = { T_i^{g+1}, if f(T_i^{g+1}) ≤ f(X_i^g)
                    { X_i^g,     otherwise
        """
        if fitness_trial <= fitness_target:
            return trial, fitness_trial
        else:
            return target, fitness_target
    
    def enforce_bounds(self, vector, bounds):
        """
        Ensure vector stays within bounds
        """
        for i in range(len(vector)):
            vector[i] = np.clip(vector[i], bounds[i][0], bounds[i][1])
        return vector
    
    def optimize(self, ann_model, fixed_params, bounds, verbose=True):
        """
        Main optimization loop
        
        Parameters:
        - ann_model: Trained ANN model for predictions
        - fixed_params: [SPP, Q, Depth, Hook_Load] - fixed parameters
        - bounds: [(min_wob, max_wob), (min_rpm, max_rpm)] - optimization bounds
        """
        # Initialize population
        population = self.initialize_population(bounds)
        
        # Evaluate initial population
        fitness = np.array([
            self.objective_function(ind, ann_model, fixed_params) 
            for ind in population
        ])
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = [best_fitness]
        
        if verbose:
            print(f"Initial best fitness: {best_fitness:.4f}")
            print(f"Initial best solution: WOB={best_solution[0]:.2f}, RPM={best_solution[1]:.2f}")
        
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
                
                # Evaluate trial vector
                fitness_trial = self.objective_function(trial, ann_model, fixed_params)
                
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
                print(f"Generation {generation + 1}: Best fitness = {best_fitness:.4f}")
        
        # Calculate final predictions
        full_params = np.concatenate([best_solution, fixed_params])
        final_rop = ann_model.predict_rop(full_params)
        final_torque = ann_model.predict_torque(full_params) * 0.737562  # Convert to Lb.Ft
        
        results = {
            'optimal_wob': best_solution[0],
            'optimal_rpm': best_solution[1],
            'predicted_rop': final_rop,
            'predicted_torque': final_torque,
            'fitness_history': fitness_history,
            'final_fitness': best_fitness
        }
        
        if verbose:
            print("\nOptimization Results:")
            print(f"Optimal WOB: {results['optimal_wob']:.2f}")
            print(f"Optimal RPM: {results['optimal_rpm']:.2f}")
            print(f"Predicted ROP: {results['predicted_rop']:.2f} m/hr")
            print(f"Predicted Torque: {results['predicted_torque']:.2f} Lb.Ft")
        
        return results
    
    def plot_convergence(self, fitness_history):
        """
        Plot optimization convergence
        """
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history)
        plt.title('Differential Evolution Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.show()
