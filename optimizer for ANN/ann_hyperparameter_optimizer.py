import numpy as np
from sklearn.model_selection import train_test_split
from drilling_ann import DrillingPredictor
from sklearn.metrics import r2_score
from de_optimizer import DifferentialEvolution

def ann_objective(params, X, y):
    """
    Objective: minimize negative R² score on validation set.
    params:
        [hidden1, hidden2, dropout, lr, batch_size]
    """
    # Round integer params where needed
    hidden1 = int(params[0])
    hidden2 = int(params[1])
    dropout_rate = np.clip(params[2], 0.0, 0.5)
    learning_rate = max(params[3], 1e-5)
    batch_size = int(params[4])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train predictor
    model = DrillingPredictor(
        hidden_sizes=[hidden1, hidden2],
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=100,
        dropout_rate=dropout_rate
    )
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    results = model.evaluate(X_val, y_val)
    return -results['r2']  # DE minimizes, so negative R²

def optimize_ann_hyperparameters(data, target_column='ROP', pop_size=10, max_iter=20):
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values
    y = data[target_column].values

    bounds = [
        (8, 64),     # hidden1
        (4, 32),     # hidden2
        (0.0, 0.5),  # dropout
        (1e-4, 1e-2),# learning rate
        (16, 64)     # batch size
    ]

    def wrapped_objective(x):
        return ann_objective(x, X, y)  # calls ANN trainer

    de = DifferentialEvolution(pop_size=pop_size, F=0.5, CR=0.7, max_iter=max_iter)

    # *** Custom loop instead of de.optimize ***
    population = de.initialize_population(bounds)
    fitness = np.array([wrapped_objective(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    for generation in range(max_iter):
        for i in range(pop_size):
            donor = de.mutate(population, i)
            donor = de.enforce_bounds(donor, bounds)
            trial = de.crossover(population[i], donor)
            trial = de.enforce_bounds(trial, bounds)
            fitness_trial = wrapped_objective(trial)
            selected, selected_fitness = de.selection(population[i], trial, fitness[i], fitness_trial)
            population[i] = selected
            fitness[i] = selected_fitness
            if selected_fitness < best_fitness:
                best_solution = selected.copy()
                best_fitness = selected_fitness

    print("\nBest ANN hyperparameters found:")
    print(best_solution)
    return best_solution

