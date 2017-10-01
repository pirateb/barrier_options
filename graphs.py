import numpy as np
import pandas as pd

from matrix import covariance_from_corr_std, corr_2_cov
from barrier import price_barrier_single
import matplotlib.pyplot as plt
from stock_movements import simulate_stock_movements_brownian
from prices import get_prices
import seaborn as sns


def create_sigma_corr_graph():
    steps = 500
    T = 1
    r = 0.05

    initial_prices = np.array([100.0, 100.0])
    strikes = np.array([100.0, 100.0])

    rho_range = np.linspace(0, 0.95, 10)
    sigma_range = np.linspace(0.05, 0.3, 10)

    runs = 50000

    def calc_surface(barrier):
        results = []
        for rho in rho_range:
            temp_r = []
            for sigma in sigma_range:
                cov = covariance_from_corr_std(rho, sigma)
                price = price_barrier_single(runs=runs, cov=cov, T=T, steps=steps,
                                             initial_prices=initial_prices, strikes=strikes, barrier=barrier,
                                             r=r)
                temp_r.append(price)
            results.append(temp_r)

        results = np.array(results)
        return results

    barrier = np.array([130.0, 130.0])
    results = calc_surface(barrier)
    for res in results:
        plt.plot(sigma_range, res)
    plt.legend(np.round(rho_range, 2))
    plt.title("Price of multiple asset barrier option (call | knockout)")
    plt.xlabel("Sigma")
    plt.ylabel("Price")
    plt.savefig('sigma_barrier.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    plt.contourf(sigma_range, rho_range, results)
    plt.colorbar()
    plt.xlabel("sigma")
    plt.ylabel("rho")
    plt.title("Barrier option price (rho vs sigma)")
    plt.savefig('sigma_barrier_contour.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    barrier = np.array([170.0, 170.0])
    results = calc_surface(barrier)
    for res in results:
        plt.plot(sigma_range, res)
    plt.legend(np.round(rho_range, 2))
    plt.title("Price of multiple asset barrier option (call | knockout)")
    plt.xlabel("Sigma")
    plt.ylabel("Price")
    plt.savefig('sigma_barrier_large.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    plt.contourf(sigma_range, rho_range, results)
    plt.colorbar()
    plt.xlabel("sigma")
    plt.ylabel("rho")
    plt.title("Barrier option price (rho vs sigma)")
    plt.savefig('sigma_barrier_large_contour.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()


def create_naive_histogram_montecarlo():
    cov = np.array([0.2 ** 2])
    cov = cov.reshape((len(cov), 1))
    steps = 500
    T = 1
    r = 0.05
    initial_prices = np.array([100.0])
    strikes = np.array([100.0])
    barrier = np.array([130.0])

    def multiple_runs(runs, iterations=200, **kwargs):
        result = []
        for _ in range(iterations):
            result.append(
                price_barrier_single(runs=runs, cov=cov, T=T, steps=steps,
                                     initial_prices=initial_prices, strikes=strikes, barrier=barrier,
                                     r=r, **kwargs)
            )

        return result

    res = []
    run_list = [10, 20, 50, 100, 250, 500, 1000, 2500, 10000]
    for run in run_list:
        print("working on {0}".format(run))
        res.append(multiple_runs(runs=run))

    plt.style.use('ggplot')
    plt.errorbar(range(len(res)), [np.mean(x) for x in res], yerr=[np.std(x) for x in res], fmt="o--")
    plt.title("Price of single barrier option (call | knockout)")
    plt.xlabel("Number of runs")
    plt.ylabel("Price")
    plt.xticks(range(len(res)), run_list, rotation='vertical')
    plt.savefig('barrier_option.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    std = [np.std(x) for x in res]
    theo_std = std[-1] * np.sqrt([run_list[-1] / x for x in run_list])
    plt.plot(theo_std, "ro--")
    plt.plot(std, "bo--")
    plt.title("Observed vs theoretical standard deviation")
    plt.legend(["Theoretical", "Observed"])
    plt.xlabel("Number of runs")
    plt.ylabel("Price")
    plt.xticks(range(len(res)), run_list, rotation='vertical')
    plt.savefig('standard_deviation.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    ## now lets create a graph with antithetic
    res_antithetic = []
    for run in run_list:
        print("working on {0}".format(run))
        res_antithetic.append(multiple_runs(runs=run, run_list="ANTITHETIC"))

    std_antithetic = [np.std(x) for x in res_antithetic]
    plt.plot(std, "ro--")
    plt.plot(std_antithetic, "bo--")
    plt.title("Standard deviation (standard vs antithetic)")
    plt.legend(["Standard", "Antithetic"])
    plt.xlabel("Number of runs")
    plt.ylabel("Price")
    plt.xticks(range(len(res)), run_list, rotation='vertical')
    plt.savefig('standard_antithetic_std.eps', bbox_inches='tight', format='eps', dpi=1000)


def simulated_stocks_examples():
    cov = np.array([0.2 ** 2])
    cov = cov.reshape((len(cov), 1))
    steps = 500
    T = 10
    r = 0.05

    initial_prices = 100
    movements = simulate_stock_movements_brownian(cov, T, steps, r)
    movements = movements.cumsum(axis=1).T

    simulated_prices = initial_prices * np.exp(movements)

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(np.linspace(0, T, steps), movements)
    axarr[0].set_title("Simulated returns")
    axarr[0].set_xlabel("Time")
    axarr[0].set_ylabel("Return")

    axarr[1].plot(np.linspace(0, T, steps), simulated_prices)
    axarr[1].set_title("Simulated price")
    axarr[1].set_xlabel("Time")
    axarr[1].set_ylabel("Price")

    plt.plot(np.linspace(0, T, steps), simulated_prices)
    plt.title("Simulated stock price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.savefig('simulated_price.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    strike = 80
    plt.plot([0, T], [strike, strike], "r--")
    plt.title("Simulated stock price with strike")
    plt.legend(["price", "strike"])
    plt.savefig('simulated_price_strike.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    barrier = 120
    plt.plot([0, T], [barrier, barrier], "b--")
    plt.title("Simulated stock price with strike and barrier")
    plt.legend(["price", "strike"])
    plt.savefig('simulated_price_strike_barrier.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    def generate_correlated_series(rho: float):
        corr = np.array([[1.0, rho], [rho, 1.0]])
        sigma = np.array([0.1, 0.1])
        initial_prices = np.array([100, 100])
        cov = corr_2_cov(corr, sigma)
        movements = simulate_stock_movements_brownian(cov, T, steps, r)
        movements = movements.cumsum(axis=1).T

        simulated_prices = initial_prices * np.exp(movements)
        return simulated_prices

    plt.plot(np.linspace(0, T, steps), generate_correlated_series(rho=0.9), "r", alpha=0.8)
    plt.plot(np.linspace(0, T, steps), generate_correlated_series(rho=0.0), "b", alpha=0.8)
    plt.title("Simulated correlated stock prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.savefig('multi_simulated_price.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()


def price_barrier_google(T: float = 1.0, r: float = 0.05, barrier_percent: float = 1.3, normalize_prices: bool = False):
    NUMBER_STOCKS_TO_ANALYSE = 5

    def get_symbols():
        df = pd.read_csv("nasdaq100list.csv")
        df = df.sort_values(by="Nasdaq100_points").tail(NUMBER_STOCKS_TO_ANALYSE)
        symbols = df["Symbol"].values
        return symbols

    symbols = get_symbols()
    df = get_prices(symbols)
    log_df = np.log(df)

    cov = log_df.cov().values

    def calculate_barrier_prices(normalize_prices: bool):
        if normalize_prices:
            initial_prices = np.ones(df.values.shape[1]) * 100.0
        else:
            initial_prices = df.values[-1, :]

        runs = 25000
        steps = 100

        barrier_prices = np.zeros(cov.shape)
        for i, symbol in enumerate(symbols):
            for j, inner_symbol in enumerate(symbols[i + 1:]):
                j += i + 1  # refactor index to normal
                print("Calculating barrier price for {0} and {1}".format(symbol, inner_symbol))
                tmp_cov = np.array([cov[i, i], cov[i, j], cov[j, i], cov[j, j]]).reshape((2, 2))
                price = price_barrier_single(runs=runs, cov=tmp_cov, T=T, steps=steps,
                                             initial_prices=np.take(initial_prices, [i, j]),
                                             strikes=np.take(initial_prices, [i, j]),
                                             barrier=np.round(np.take(initial_prices * barrier_percent, [i, j])),
                                             r=r)
                print(price)
                barrier_prices[i, j] = price
                barrier_prices[j, i] = price

        return barrier_prices

    # create various graphs - normal barrier prices
    barrier_prices = calculate_barrier_prices(normalize_prices=False)
    sns.heatmap(pd.DataFrame(barrier_prices,columns=symbols, index=symbols)
                , annot=True
                , linewidths=.5)
    plt.title("Two asset call barrier (knock out) option price")
    plt.savefig('barrier_heatmap_nasdaq.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    barrier_prices = calculate_barrier_prices(normalize_prices=True)
    sns.heatmap(pd.DataFrame(barrier_prices,columns=symbols, index=symbols)
                , annot=True
                , linewidths=.5)
    plt.title("Two asset call barrier (knock out) option price. Normalized price.")
    plt.savefig('barrier_heatmap_nasdaq_normalized.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()

    sns.heatmap(pd.DataFrame(log_df.cov(),columns=symbols, index=symbols)
                , annot=True
                , linewidths=.5)
    plt.title("Covariance matrix for top NASDAQ stocks")
    plt.savefig('barrier_heatmap_cov.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.clf()