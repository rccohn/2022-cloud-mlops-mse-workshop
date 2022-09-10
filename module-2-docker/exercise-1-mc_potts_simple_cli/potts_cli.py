from argparse import ArgumentParser
from mc_potts import mc_potts


def parse_inputs():
    """
    Read user input from command line.
    """
    parser = ArgumentParser(description="Run Monte Carlo Potts model "\
                "and return resulting state in space/newline separated values.")
    
    parser.add_argument("-n", dest="n_sites", action="store", default=50,
                        help="Length of square system, pixels.")
    parser.add_argument("-t", dest="n_steps", action="store", default=25,
                        help="Number of MC time steps.")
    parser.add_argument("--kT", dest="kT", action="store", default=0.9,
                        help="Boltzmann constant k times temperature T")
    parser.add_argument("-s", dest="seed", action="store", default=None,
                        help="Random seed, optional.")
    
    return parser.parse_args()

def main():
    # get arguments from command line
    args = parse_inputs()
    # run simulation
    x = mc_potts(int(args.n_sites), int(args.n_steps), float(args.kT), 
        int(args.seed) if args.seed is not None else None)
    
    # return results as csv string
    for row in x:
        print(" ".join([str(x) for x in row]), end="\n")


if __name__ == "__main__":
    main()
