import numpy as np
from sklearn.model_selection import train_test_split
from drilling_ann import DrillingPredictor
from sklearn.metrics import r2_score
from de_optimizer import DifferentialEvolution
from data_augmentation import augment_training_data  # ADD THIS IMPORT

def ann_objective(params, X, y):
    """
    Objective: minimize negative R² score on validation set.
    params:
        [hidden_neurons, dropout, lr_index, batch_size, weight_decay]
    """
    # Round integer params where needed
    hidden_neurons = int(params[0])
    dropout_rate = np.clip(params[1], 0.2, 0.5)

    # Map lr_index to one of three valid learning rates
    valid_lrs = [0.01, 0.001]
    lr_index = int(np.clip(params[2], 0, 2))
    learning_rate = valid_lrs[lr_index]

    batch_size = int(params[3])
    weight_decay = params[4]

    # Split data first
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # AUGMENT TRAINING DATA (not validation!)
    X_train_aug, y_train_aug = augment_training_data(
        X_train, y_train,
        noise_levels=[0.10],
        verbose=False  # Silent during optimization
    )

    # Train predictor with single hidden layer using AUGMENTED data
    model = DrillingPredictor(
        hidden_sizes=[hidden_neurons],  # Single hidden layer
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=100,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay
    )
    model.train(X_train_aug, y_train_aug, X_val, y_val)

    # Evaluate on ORIGINAL validation data
    results = model.evaluate(X_val, y_val)
    return -results['r2']  # DE minimizes, so negative R²

def optimize_ann_hyperparameters(data, target_column='ROP', pop_size=10, max_iter=20):
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values
    y = data[target_column].values

    # Updated bounds for single hidden layer
    bounds = [
        (4, 16),     # hidden_neurons - single layer
        (0.2, 0.5),  # dropout
        (0, 1),      # lr_index (0=0.001, 1=0.0005, 2=0.0001)
        (2, 8),      # batch_size
        (1e-5, 1e-3) # weight_decay
    ]

    def wrapped_objective(x):
        return ann_objective(x, X, y)

    de = DifferentialEvolution(pop_size=pop_size, F=0.5, CR=0.7, max_iter=max_iter)

    # Custom loop
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

    # Decode learning rate
    valid_lrs = [0.001, 0.0005, 0.0001]
    lr_index = int(np.clip(best_solution[2], 0, 2))
    final_lr = valid_lrs[lr_index]

    print("\nBest ANN hyperparameters found:")
    print(f"Hidden neurons: {int(best_solution[0])}")
    print(f"Dropout: {best_solution[1]:.3f}")
    print(f"Learning rate: {final_lr}")
    print(f"Batch size: {int(best_solution[3])}")
    print(f"Weight decay: {best_solution[4]:.5f}")
    print(f"Best R²: {-best_fitness:.4f}")

    return best_solution