scenarios = {
    1: {
        "price": [30.24, 18.14, 11.49],
        "lambda_dict": {30.24: 23.314, 18.14: 30.843, 11.49: 38.371},
        "mu_dict": {30.24: 1/26, 18.14: 1/26, 11.49: 1/26},
    },
    2: {
        "price": [30.24, 18.14, 11.49],
        "lambda_dict": {30.24: 23.314, 18.14: 30.843, 11.49: 38.371},
        "mu_dict": {30.24: 1/4, 18.14: 1/26, 11.49: 1/52},
    },
    3: {
        "price": [11.79, 7.08, 4.48],
        "lambda_dict": {11.79: 13.123, 7.08: 16.011, 4.48: 18.900},
        "mu_dict": {11.79: 1/26, 7.08: 1/26, 4.48: 1/26},
    },
    4: {
        "price": [11.79, 7.08, 4.48],
        "lambda_dict": {11.79: 13.123, 7.08: 16.011, 4.48: 18.900},
        "mu_dict": {11.79: 1/4, 7.08: 1/26, 4.48: 1/52},
    },
}

# Experimental parameters
T = 260  # Planning horizon (260 weeks as mentioned in paper)
I_0 = 0  # Initial physical inventory level (10×10³ units as mentioned)
c_a = 1000   # Acquisition cost ($1000 as mentioned)
c_h = 10 # Holding cost ($10 as mentioned)
gamma = 0.99  # Discount factor
max_acquisition = 10  # Maximum acquisition per period (10×10³ as mentioned)


