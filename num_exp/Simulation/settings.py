scenarios = {
    1: {
        "price": [18,16,14,12,10],
        "lambda_dict": {18:5.5, 16:6, 14:6.5, 12:8, 10:9.5},
        "mu_dict": {18:0.1, 16:0.1, 14:0.1, 12:0.1, 10:0.1},
    },
    2: {
        "price": [18,16,14,12,10],
        "lambda_dict": {18:5.2, 16:6, 14:7.5, 12:8.5, 10:9.5},
        "mu_dict": {18:0.1, 16:0.1, 14:0.1, 12:0.1, 10:0.1},
    },
    3: {
        "price": [18,16,14,12,10],
        "lambda_dict": {18:5, 16:5.5, 14:6, 12:7.5, 10:9.8},
        "mu_dict": {18:0.1, 16:0.1, 14:0.1, 12:0.1, 10:0.1},
    },
    4: {
        "price": [18,16,14,12,10],
        "lambda_dict": {18:5.5, 16:6, 14:6.5, 12:8, 10:9.5},
        "mu_dict": {18:0.08, 16:0.09, 14:0.1, 12:0.11, 10:0.12},
    },
    5: {
        "price": [18,16,14,12,10],
        "lambda_dict": {18:5.2, 16:6, 14:7.5, 12:8.5, 10:9.5},
        "mu_dict": {18:0.06, 16:0.08, 14:0.1, 12:0.12, 10:0.14},
    },
    6: {
        "price": [18,16,14,12,10],
        "lambda_dict": {18:5, 16:5.5, 14:6, 12:7.5, 10:9.8},
        "mu_dict": {18:0.08, 16:0.09, 14:0.1, 12:0.11, 10:0.12},
    }
}

# Experimental parameters
T = 1000                # Planning horizon
I_0 = 0                 # Initial physical inventory level
c_a = 1000              # Acquisition cost
c_h = 10                # Holding cost
gamma = 0.99            # Discount factor
max_acquisition = 10    # Maximum acquisition per period


