import numpy as np
from stock_movements import simulate_stock_movements_brownian


def price_barrier_single_run(cov, T, steps, initial_prices, strikes, barrier, r):
    movements = simulate_stock_movements_brownian(cov, T, steps, r)
    movements = movements.cumsum(axis=1).T

    simulated_prices = initial_prices * np.exp(movements)

    # option price is zero for any stock that hit barrier
    barrier_breach = np.any(simulated_prices > barrier, axis=0)
    dimensions = cov.shape[0]
    multiplier = np.ones(dimensions)
    multiplier[barrier_breach] = 0

    # else its the option value
    call_value = np.maximum((simulated_prices[-1, :] - strikes), 0)
    discount_factor = np.exp(-r * T)
    barrier_options = discount_factor * np.multiply(call_value, multiplier)
    return np.min(barrier_options)


def price_barrier_single_run_antithetic(cov, T, steps, initial_prices, strikes, barrier, r):
    steps = steps / 2
    movements = simulate_stock_movements_brownian(cov, T, steps, r)
    movements = movements.cumsum(axis=1).T

    movements_2 = simulate_stock_movements_brownian(cov, T, steps, r)
    movements_2 = movements_2.cumsum(axis=1).T

    movements = (movements + movements_2)/2

    def price_barrier(movements):
        simulated_prices = initial_prices * np.exp(movements)

        # option price is zero for any stock that hit barrier
        barrier_breach = np.any(simulated_prices > barrier, axis=0)
        dimensions = cov.shape[0]
        multiplier = np.ones(dimensions)
        multiplier[barrier_breach] = 0

        # else its the option value
        call_value = np.maximum((simulated_prices[-1, :] - strikes), 0)
        discount_factor = np.exp(-r * T)
        barrier_options = discount_factor * np.multiply(call_value, multiplier)
        return np.min(barrier_options)

    return 0.5 * (price_barrier(movements))


def price_barrier_single(runs, cov, T, steps, initial_prices, strikes, barrier, r, **kwargs):
    import multiprocessing
    from itertools import repeat
    if "run_method" in kwargs:
        run_method = kwargs["run_method"]
    else:
        run_method = "NORMAL"

    RUN_METHODS = ["NORMAL", "ANTITHETIC"]
    if run_method not in RUN_METHODS:
        raise ValueError("Unknown run method {0}".format(run_method))

    # create a pool with the number of threads this pc has
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    np.random.seed()

    if run_method == "NORMAL":
        results = pool.starmap(price_barrier_single_run,  # function to run
                               repeat((cov, T, steps, initial_prices, strikes, barrier, r), runs)
                               # arguments for each run
                               )
    elif run_method == "ANTITHETIC":
        results = pool.starmap(price_barrier_single_run_antithetic,  # function to run
                               repeat((cov, T, steps, initial_prices, strikes, barrier, r), runs)
                               # arguments for each run
                               )
    else:
        raise RuntimeError("Run method {0} not handled".format(run_method))

    pool.close()
    return np.mean(results)
