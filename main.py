from graphs import create_sigma_corr_graph, create_naive_histogram_montecarlo, simulated_stocks_examples, price_barrier_google

# TODO: Follow heston model for dynamics
# http://www.ann.ece.ufl.edu/courses/eel6686_14spr/slides/Ishan_Dalal_Short_Ppt_Multi-Barrier_Options.pdf


def main():
    # create all graphs that are required for presentation

    # sensitivity analysis of call knock out barrier option
    print("Creating sensitivity graphs")
    create_sigma_corr_graph()

    print("Creating montecarlo sensitivity graphs")
    # demonstration of clt in montecarlo pricing
    create_naive_histogram_montecarlo()

    print("Creating simulated stock paths")
    # demonstrate stock paths
    simulated_stocks_examples()

    print("Pricing options based off google data")
    # price the top 5 nasdaq stocks using google data
    price_barrier_google()

if __name__ == "__main__":
    main()