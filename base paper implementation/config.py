# Configuration parameters for drilling optimization
class Config:
    # ANN Model Parameters
    ROP_HIDDEN_NEURONS = 23  # As optimized in the paper
    TORQUE_HIDDEN_NEURONS = 27  # As optimized in the paper
    ACTIVATION_FUNCTION = 'tanh'  # Tansig function
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 500
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 50
    
    # Data Processing Parameters
    TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
    OUTLIER_THRESHOLD = 1.5  # IQR multiplier for outlier removal
    
    # Differential Evolution Parameters
    DE_POPULATION_SIZE = 30
    DE_MUTATION_FACTOR = 0.5  # F parameter
    DE_CROSSOVER_RATE = 0.7  # CR parameter
    DE_MAX_ITERATIONS = 100
    
    # Drilling Parameters
    TORQUE_MIN_LIMIT = 13000  # Lb.Ft
    TORQUE_MAX_LIMIT = 19000  # Lb.Ft
    TORQUE_PENALTY_WEIGHT = 0.1
    
    # Optimization Bounds
    WOB_MIN = 5.0   # tons
    WOB_MAX = 25.0  # tons
    RPM_MIN = 60.0
    RPM_MAX = 200.0
    
    # Drilling Simulation
    STAND_LENGTH = 30.0  # meters per stand
    STANDS_PER_UPDATE = 4  # Model update frequency
    
    # Feature columns (must match your data)
    FEATURE_COLUMNS = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    TARGET_COLUMNS = ['ROP', 'Surface_Torque']
