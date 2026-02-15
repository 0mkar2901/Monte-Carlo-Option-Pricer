import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 100

class AdvancedMonteCarloEngine:
    """
    Professional-grade Monte Carlo Engine for Pricing Path-Dependent Options.
    Refined for Business Time (252 days) and 1/sqrt(N) error analysis.
    """
    def __init__(self, S0, K, T, r, sigma, iterations=100000):
        self.S0 = 400
        self.K = S0+20
        self.T = T
        self.r = 0.15
        self.sigma = 0.80
        self.iterations = iterations

        # PILLAR 1: Business Time Convention (252 trading days per year)
        self.steps = int(T * 252)
        self.dt = T / self.steps
        self.paths = None

    def simulate_paths(self):
        """Generates vectorized price paths using GBM with Ito's Lemma correction."""
        Z = np.random.standard_normal((self.iterations, self.steps))

        # PILLAR 2: Ito's Lemma Drift Correction (-0.5 * sigma^2)
        # This accounts for the 'Volatility Drag' discussed in our notes
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z

        log_returns = drift + diffusion
        path_cumulative_returns = np.cumsum(log_returns, axis=1)

        self.paths = self.S0 * np.exp(path_cumulative_returns)
        self.paths = np.insert(self.paths, 0, self.S0, axis=1)
        return self.paths

    def calculate_precision(self, payoffs):
        """
        PILLAR 3: The 1/sqrt(N) Error Metric
        Calculates the statistical confidence of the simulation.
        """
        # Standard Error of the Mean (SEM)
        std_err = np.std(payoffs) / np.sqrt(self.iterations)
        # Discounted to Present Value
        return std_err * np.exp(-self.r * self.T)

    def price_asian_option(self, option_type='call'):
        """Prices an Arithmetic Asian Option based on the path average."""
        if self.paths is None: self.simulate_paths()

        path_averages = np.mean(self.paths, axis=1)

        if option_type == 'call':
            payoffs = np.maximum(path_averages - self.K, 0)
        else:
            payoffs = np.maximum(self.K - path_averages, 0)

        price = np.mean(payoffs) * np.exp(-self.r * self.T)

        # Apply the 1/sqrt(N) precision logic
        error = self.calculate_precision(payoffs)
        conf_interval = (price - 1.96 * error, price + 1.96 * error)

        return price, conf_interval

    # --- VISUALIZATION METHODS (KEEPING YOUR ANIMATION LOGIC) ---
    def plot_simulation_animated(self, n_paths=25, total_frames=100):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(13, 6.5), dpi=90)
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#0a0a0a')

        colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_paths))
        lines = [ax.plot([], [], lw=1.8, alpha=0.9, color=colors[i])[0] for i in range(n_paths)]
        ax.axhline(self.K, color='#00ff41', linestyle='--', linewidth=3, label=f'Strike K=${self.K}')

        progress_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, color='#00ffff',
                               bbox=dict(boxstyle='round', facecolor='#000000', edgecolor='#00ffff', alpha=0.85))

        ax.set_xlim(0, self.steps)
        ax.set_ylim(np.min(self.paths[:n_paths]) * 0.92, np.max(self.paths[:n_paths]) * 1.08)

        frame_steps = np.linspace(0, self.steps, total_frames, dtype=int)

        def animate(frame):
            step = frame_steps[frame]
            for i, line in enumerate(lines):
                line.set_data(np.arange(step + 1), self.paths[i, :step + 1])
            progress = (step / self.steps) * 100
            progress_text.set_text(f'Day {step}/{self.steps} │ {progress:.0f}% Complete')
            return lines + [progress_text]

        anim = FuncAnimation(fig, animate, frames=total_frames, interval=50, blit=True)
        plt.close(fig)
        return HTML(anim.to_jshtml().replace('loop controls', 'loop autoplay muted'))

    def plot_convergence(self, option_type='call'):
        """Visualizes how price stabilizes as 1/sqrt(N) takes effect."""
        path_averages = np.mean(self.paths, axis=1)
        payoffs = np.maximum(path_averages - self.K, 0) if option_type == 'call' else np.maximum(self.K - path_averages, 0)
        discount = np.exp(-self.r * self.T)

        sim_range = np.arange(100, self.iterations, 500)
        prices = [np.mean(payoffs[:n]) * discount for n in sim_range]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(sim_range, prices, color='#00ff41', lw=2, label='Simulated Price')
        ax.axhline(prices[-1], color='#ff00ff', ls='--', label=f'Converged: ${prices[-1]:.4f}')
        ax.set_title("Mathematical Convergence via Law of Large Numbers")
        ax.set_xlabel("Number of Simulations (N)")
        ax.set_ylabel("Option Price ($)")
        ax.legend()
        plt.show()

# ═══════════════════════════════════════════════════════════
#  EXECUTION
# ═══════════════════════════════════════════════════════════

# Initialize with conceptual parameters
engine = AdvancedMonteCarloEngine(S0=100, K=105, T=1.0, r=0.05, sigma=0.25, iterations=100000)

print(" MATH CHECK:")
print(f"Using {engine.steps} steps based on 252 trading days.")
print(f"Drift correction (-0.5 * sigma^2) applied: {-0.5 * engine.sigma**2:.4f}")

engine.simulate_paths()
price, conf = engine.price_asian_option()

print(f"\n FINAL FAIR VALUE: ${price:.4f}")
print(f" STATISTICAL PRECISION (1/sqrt(N)): ±${(conf[1]-price):.4f}")

# Display visualizations
display(engine.plot_simulation_animated())
engine.plot_convergence()
