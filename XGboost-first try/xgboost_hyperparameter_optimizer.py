# xgboost_hyperparameter_optimizer.py
import numpy as np
from sklearn.model_selection import train_test_split
from drilling_xgboost import DrillingXGBoostPredictor
from sklearn.metrics import r2_score
from de_optimizer import DifferentialEvolution

def xgboost_objective(params, X, y):
    """
    Objective: minimize negative R² score on validation set.

    Parameters:
    -----------
    params : array-like
        [n_estimators, max_depth, learning_rate, subsample,
         colsample_bytree, gamma, min_child_weight, reg_alpha, reg_lambda]
    """
    # Extract and process parameters
    n_estimators = int(params[0])
    max_depth = int(params[1])
    learning_rate = params[2]
    subsample = np.clip(params[3], 0.5, 1.0)
    colsample_bytree = np.clip(params[4], 0.5, 1.0)
    gamma = max(0, params[5])
    min_child_weight = int(max(1, params[6]))
    reg_alpha = max(0, params[7])
    reg_lambda = max(0, params[8])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost model
    model = DrillingXGBoostPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )

    # FIX: Remove verbose=False parameter
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    results = model.evaluate(X_val, y_val)
    return -results['r2']  # DE minimizes, so negative R²

def optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val,
                                     pop_size=20, max_iter=30):
    """
    Optimize XGBoost hyperparameters using Differential Evolution

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

    # Define bounds for XGBoost hyperparameters
    bounds = [
        (50, 300),      # n_estimators
        (3, 10),        # max_depth
        (0.01, 0.3),    # learning_rate
        (0.5, 1.0),     # subsample
        (0.5, 1.0),     # colsample_bytree
        (0, 5),         # gamma
        (1, 10),        # min_child_weight
        (0, 1.0),       # reg_alpha (L1)
        (0, 2.0)        # reg_lambda (L2)
    ]

    def wrapped_objective(x):
        return xgboost_objective(x, X, y)

    print(f"\n{'=' * 70}")
    print(f"Starting XGBoost Hyperparameter Optimization")
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

    print(f"\n{'=' * 70}")
    print(f"Optimization Complete")
    print(f"{'=' * 70}")
    print("\nBest XGBoost hyperparameters found:")
    print(f"  n_estimators:      {int(best_solution[0])}")
    print(f"  max_depth:         {int(best_solution[1])}")
    print(f"  learning_rate:     {best_solution[2]:.4f}")
    print(f"  subsample:         {best_solution[3]:.4f}")
    print(f"  colsample_bytree:  {best_solution[4]:.4f}")
    print(f"  gamma:             {best_solution[5]:.4f}")
    print(f"  min_child_weight:  {int(best_solution[6])}")
    print(f"  reg_alpha (L1):    {best_solution[7]:.4f}")
    print(f"  reg_lambda (L2):   {best_solution[8]:.4f}")
    print(f"\n  Best R²:           {-best_fitness:.4f}")
    print(f"{'=' * 70}\n")

    # Return as dictionary
    best_params = {
        'n_estimators': int(best_solution[0]),
        'max_depth': int(best_solution[1]),
        'learning_rate': best_solution[2],
        'subsample': best_solution[3],
        'colsample_bytree': best_solution[4],
        'gamma': best_solution[5],
        'min_child_weight': int(best_solution[6]),
        'reg_alpha': best_solution[7],
        'reg_lambda': best_solution[8]
    }

    return best_params, history
