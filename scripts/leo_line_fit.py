import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleProbabilitySplineFitter:
    def __init__(self, probability_slices):
        """
        Parameters:
        -----------
        probability_slices : np.ndarray
            Shape (n_slices, height, width)
        """
        self.slices = probability_slices
        self.n_slices = probability_slices.shape[0]
        self.height = probability_slices.shape[1]
        self.width = probability_slices.shape[2]

    def get_probability(self, z_idx, x, y):
        """Bilinear interpolation of probability at (x, y) in slice z_idx."""
        x = np.clip(x, 0, self.width - 1.001)
        y = np.clip(y, 0, self.height - 1.001)

        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)

        wx, wy = x - x0, y - y0

        return ((1-wx)*(1-wy)*self.slices[z_idx, y0, x0] +
                wx*(1-wy)*self.slices[z_idx, y0, x1] +
                (1-wx)*wy*self.slices[z_idx, y1, x0] +
                wx*wy*self.slices[z_idx, y1, x1])

    def spline_from_params(self, params, n_control):
        """Build spline from flattened control point parameters."""
        xy = params.reshape(n_control, 2)
        z = np.linspace(0, self.n_slices - 1, n_control)

        k = min(3, n_control - 1)
        tck, _ = splprep([xy[:, 0], xy[:, 1], z], s=0, k=k)
        return tck

    def evaluate_at_slices(self, tck):
        """Get (x, y) coordinates where spline intersects each slice."""
        # Sample spline densely
        u = np.linspace(0, 1, 1000)
        x, y, z = splev(u, tck)

        # Find closest point to each slice
        points = np.zeros((self.n_slices, 2))
        for i in range(self.n_slices):
            idx = np.argmin(np.abs(z - i))
            points[i] = [x[idx], y[idx]]

        return points

    def objective(self, params, n_control):
        """Negative sum of log probabilities (to minimize)."""
        try:
            tck = self.spline_from_params(params, n_control)
            points = self.evaluate_at_slices(tck)

            log_prob_sum = 0
            for i in range(self.n_slices):
                prob = self.get_probability(i, points[i, 0], points[i, 1])
                log_prob_sum += np.log(prob + 1e-10)

            return -log_prob_sum
        except:
            return 1e10

    def fit(self, min_control=3, max_control=None, tol=0.01, verbose=True):
        """
        Fit spline with adaptive number of control points.

        Parameters:
        -----------
        min_control : int
            Starting number of control points
        max_control : int
            Maximum control points (default: n_slices)
        tol : float
            Stop when improvement < tol
        verbose : bool
            Print progress
        """
        if max_control is None:
            max_control = self.n_slices

        # Initial guess: max probability in each slice, then interpolate
        init_points = np.zeros((self.n_slices, 2))
        for i in range(self.n_slices):
            y_max, x_max = np.unravel_index(np.argmax(self.slices[i]),
                                           self.slices[i].shape)
            init_points[i] = [x_max, y_max]

        history = []
        best_result = None
        prev_score = -np.inf

        for n in range(min_control, max_control + 1):
            # Interpolate initial guess to n control points
            z_all = np.arange(self.n_slices)
            z_control = np.linspace(0, self.n_slices - 1, n)
            x_init = np.interp(z_control, z_all, init_points[:, 0])
            y_init = np.interp(z_control, z_all, init_points[:, 1])
            x0 = np.column_stack([x_init, y_init]).flatten()

            # Bounds
            bounds = [(0, self.width-1), (0, self.height-1)] * n

            # Optimize using L-BFGS-B (much faster than differential_evolution)
            result = minimize(
                fun=lambda p: self.objective(p, n),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            score = -result.fun
            improvement = score - prev_score

            tck = self.spline_from_params(result.x, n)
            control_points = result.x.reshape(n, 2)

            history.append({
                'n_control': n,
                'score': score,
                'improvement': improvement,
                'tck': tck,
                'control_points': control_points
            })

            if verbose:
                print(f"n={n:2d} | score={score:7.3f} | improvement={improvement:6.3f}")

            if best_result is None or score > best_result['score']:
                best_result = history[-1]

            # Check convergence
            if n > min_control and improvement < tol:
                if verbose:
                    print(f"Converged (improvement {improvement:.4f} < {tol})")
                break

            prev_score = score

            # Use current result as warm start for next iteration
            init_points = self.evaluate_at_slices(tck)

        return {
            'best_tck': best_result['tck'],
            'best_score': best_result['score'],
            'best_n': best_result['n_control'],
            'history': history
        }

    def visualize_all_slices(self, tck, figsize=(20, 12)):
        """Show spline intersection on all slices."""
        points = self.evaluate_at_slices(tck)

        n_cols = min(5, self.n_slices)
        n_rows = int(np.ceil(self.n_slices / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        for i in range(self.n_slices):
            ax = axes[i]
            ax.imshow(self.slices[i], cmap='hot', origin='lower')

            x, y = points[i]
            ax.plot(x, y, 'go', markersize=12, markeredgecolor='white',
                   markeredgewidth=2)

            prob = self.get_probability(i, x, y)
            ax.text(0.05, 0.95, f'P={prob:.3f}', transform=ax.transAxes,
                   color='white', fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

            ax.set_title(f'Slice {i}', fontsize=10)
            ax.axis('off')

        for i in range(self.n_slices, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def visualize_summary(self, results, figsize=(15, 5)):
        """Show 3D view and convergence."""
        fig = plt.figure(figsize=figsize)

        tck = results['best_tck']

        # 3D plot
        ax1 = fig.add_subplot(131, projection='3d')
        u = np.linspace(0, 1, 500)
        x, y, z = splev(u, tck)
        ax1.plot(x, y, z, 'b-', linewidth=2, label='Spline')

        # Control points
        cp = results['history'][-1]['control_points']
        z_cp = np.linspace(0, self.n_slices - 1, len(cp))
        ax1.scatter(cp[:, 0], cp[:, 1], z_cp, c='red', s=100,
                   edgecolors='black', linewidths=1.5, label=f'{len(cp)} control pts')

        # Slice intersections
        points = self.evaluate_at_slices(tck)
        z_slices = np.arange(self.n_slices)
        ax1.scatter(points[:, 0], points[:, 1], z_slices,
                   c='green', s=40, alpha=0.7, label='Intersections')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Spline')
        ax1.legend()

        # Convergence
        ax2 = fig.add_subplot(132)
        n_vals = [h['n_control'] for h in results['history']]
        scores = [h['score'] for h in results['history']]
        ax2.plot(n_vals, scores, 'o-', linewidth=2, markersize=8)
        ax2.axvline(results['best_n'], color='r', linestyle='--',
                   label=f"Best: n={results['best_n']}")
        ax2.set_xlabel('Control Points')
        ax2.set_ylabel('Total Log Probability')
        ax2.set_title('Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Probability per slice
        ax3 = fig.add_subplot(133)
        probs = [self.get_probability(i, points[i, 0], points[i, 1])
                for i in range(self.n_slices)]
        ax3.plot(probs, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Slice')
        ax3.set_ylabel('Probability')
        ax3.set_title('Probability at Intersections')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig



# Demo
def create_demo_data():
    """Create test data."""
    n_slices = 12
    height, width = 60, 60
    slices = np.zeros((n_slices, height, width))

    np.random.seed(42)
    for i in range(n_slices):
        # Main blob following a curve
        cx = 30 + 12 * np.sin(i / 1.2) + np.random.randn(1) * width * 0.05
        cy = 30 + 8 * np.cos(i / 2.5) + np.random.randn(1) * height * 0.05

        y, x = np.ogrid[:height, :width]
        gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / 32)

        # Add noise and secondary blob
        noise = np.random.randn(height, width) * 0.05
        cx2 = cx + np.random.randn() * 8
        cy2 = cy + np.random.randn() * 8
        gaussian2 = 0.3 * np.exp(-((x - cx2)**2 + (y - cy2)**2) / 32)

        slices[i] = np.maximum(gaussian + gaussian2 + noise, 0)
        slices[i] /= slices[i].sum()

    return slices



if __name__ == "__main__":
    print("Creating demo data...")
    slices = create_demo_data()

    print("\nFitting spline...\n")
    fitter = SimpleProbabilitySplineFitter(slices)
    results = fitter.fit(min_control=3, max_control=10, tol=0.01)

    print(f"\nBest: {results['best_n']} control points, "
          f"log prob = {results['best_score']:.3f}")

    # Visualize
    fig1 = fitter.visualize_all_slices(results['best_tck'])
    fig1.suptitle('Spline Intersections on All Slices', fontsize=14, y=0.98)

    fig2 = fitter.visualize_summary(results)

    plt.show()