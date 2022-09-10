from mc_potts import mc_potts
import numpy as np
from yaml import safe_load

def main():
    # load inputs
    with open("/mnt/inputs.yaml", "r") as f:
        inputs = safe_load(f)
    # run simulation with desired parameters
    results = mc_potts(**inputs)
    # save results
    np.savez("/mnt/outputs/results.npz", 
                final_state=results[0], 
                grain_sizes=results[1],
                time_steps=results[2])
    if len(results) == 3:
        # no animation, all done!
        return
    #  generate animation
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    def animate_wrapper(seq=results[3], seed=inputs['seed']):
        # shuffle grain colors so they appear more clearly
        # in animation
        n2 = np.prod(seq[0].shape)
        mp = np.arange(n2, dtype=int) + n2 # offset so there isn't overlap
                                     # during transformation
        rng = np.random.default_rng(seed)
        rng.shuffle(mp)

        # generate figure 
        fig, ax = plt.subplots(figsize=(4,4), dpi=150)
        template = seq[0].copy()
        def animate(i):
            # optional, apply color mapping for visual clarity
            for j, t in enumerate(mp):
                template[seq[i]==j] = t

            ax.clear()
            ax.imshow(template, cmap='Pastel1')
            return
        return fig, animate
    fig, fn = animate_wrapper()
    ani = FuncAnimation(fig, fn, frames=len(results[3]), repeat=False)
    ani.save("/mnt/outputs/grains.gif", dpi=150, writer=PillowWriter(fps=3))




if __name__ == "__main__":
    main()
