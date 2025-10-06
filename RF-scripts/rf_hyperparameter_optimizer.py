# rf_hyperparameter_optimizer.py
import numpy as np
from sklearn.model_selection import train_test_split
from drilling_random_forest import DrillingRandomForestPredictor
from sklearn.metrics import r2_score
from de_optimizer import DifferentialEvolution

def rf_objective(params, X, y):
    """
    Objective: minimize negative R² score on validation set.

    Parameters:
    -----------
    params : array-like
        [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features_idx]
    """
    # Extract and process parameters
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(max(2, params[2]))
    min_samples_leaf = int(max(1, params[3]))
    
    # max_features options: ['sqrt', 'log2', 0.5, 0.7, 0.9]
    max_features_options = ['sqrt', 'log2', 0.5, 0.7, 0.9]
    max_features_idx = int(np.clip(params[4], 0, len(max_features_options) - 1))
    max_features = max_features_options[max_features_idx]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    model = DrillingRandomForestPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        oob_score=False,  # Disable for faster training during optimization
        random_state=42,
        n_jobs=-1
    )

    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    results = model.evaluate(X_val, y_val)
    return -results['r2']  # DE minimizes, so negative R²

def optimize_rf_hyperparameters(X_train, y_train, X_val, y_val,
                                pop_size=20, max_iter=30):
    """
    Optimize Random Forest hyperparameters using Differential Evolution

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_val : array-like
        Validation features
    y_val : array-like
        Validation target
    pop_size : int
        Population size for DE
    max_iter : int
        Maximum iterations for DE

    Returns:
    --------
    best_params : dict
        Dictionary of optimized hyperparameters
    history : dict
        Optimization history
    """
    # Combine train and val for optimization
    X = np.vstack([X_train, X_val])
    y = np.hstack([y_train, y_val])

    # Define bounds for Random Forest hyperparameters
    bounds = [
        (50, 300),      # n_estimators
        (5, 30),        # max_depth
        (2, 20),        # min_samples_split
        (1, 10),        # min_samples_leaf
        (0, 4)          # max_features index (0-4 for 5 options)
    ]

    def wrapped_objective(x):
        return rf_objective(x, X, y)

    print(f"\n{'=' * 70}")
    print(f"Starting Random Forest Hyperparameter Optimization")
    print(f"{'=' * 70}")
    print(f"Population Size: {pop_size}")
    print(f"Max Iterations: {max_iter}")
    print(f"{'=' * 70}\n")

    de = DifferentialEvolution(pop_size=pop_size, F=0.5, CR=0.7, max_iter=max_iter)

    # Custom optimization loop with progress tracking
    population = de.initialize_population(bounds)
    fitness = np.array([wrapped_objective(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    # Track history
    history = {
        'iteration': [],
        'best_fitness': [],
        'mean_fitness': []
    }

    print(f"Initial Best R²: {-best_fitness:.4f}\n")

    for generation in range(max_iter):
        for i in range(pop_size):
            donor = de.mutate(population, i)
            donor = de.enforce_bounds(donor, bounds)
            trial = de.crossover(population[i], donor)
            trial = de.enforce_bounds(trial, bounds)
            fitness_trial = wrapped_objective(trial)
            selected, selected_fitness = de.selection(
                population[i], trial, fitness[i], fitness_trial
            )
            population[i] = selected
            fitness[i] = selected_fitness

            if selected_fitness < best_fitness:
                best_solution = selected.copy()
                best_fitness = selected_fitness

        # Record history
        history['iteration'].append(generation + 1)
        history['best_fitness'].append(best_fitness)
        history['mean_fitness'].append(np.mean(fitness))

        if (generation + 1) % 5 == 0:
            print(f"Generation {generation + 1}/{max_iter}: Best R² = {-best_fitness:.4f}")

    # Convert max_features index to actual value
    max_features_options = ['sqrt', 'log2', 0.5, 0.7, 0.9]
    max_features_idx = int(np.clip(best_solution[4], 0, len(max_features_options) - 1))
    max_features = max_features_options[max_features_idx]

    print(f"\n{'=' * 70}")
    print(f"Optimization Complete")
    print(f"{'=' * 70}")
    print("\nBest Random Forest hyperparameters found:")
    print(f"  n_estimators:      {int(best_solution[0])}")
    print(f"  max_depth:         {int(best_solution[1])}")
    print(f"  min_samples_split: {int(best_solution[2])}")
    print(f"  min_samples_leaf:  {int(best_solution[3])}")
    print(f"  max_features:      {max_features}")
    print(f"\n  Best R²:           {-best_fitness:.4f}")
    print(f"{'=' * 70}\n")

    # Return as dictionary
    best_params = {
        'n_estimators': int(best_solution[0]),
        'max_depth': int(best_solution[1]),
        'min_samples_split': int(best_solution[2]),
        'min_samples_leaf': int(best_solution[3]),
        'max_features': max_features
    }

    return best_params, history
