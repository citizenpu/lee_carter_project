import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')

def create_comprehensive_convergence_demo():
    """Create the same analysis with comprehensive visualizations"""

    # Generate the data (same as before)
    np.random.seed(42)
    years = np.arange(2000, 2020)
    ages = np.arange(0, 100)

    data = []
    for year in years:
        for age in ages:
            a_x = -4.0 + 0.02 * age - 0.0001 * age**2
            b_x = 0.001 * np.exp(-age / 25)
            k_t = -0.5 + 0.05 * (year - 2000)
            log_mortality = a_x + b_x * k_t

            mortality_rate = np.exp(log_mortality)
            mortality_rate = np.clip(mortality_rate, 0.00001, 0.1)

            data.append({
                'year': year,
                'age': age,
                'deaths': max(1, np.random.poisson(1000 * mortality_rate)),
                'exposure': max(100, 10000 + np.random.normal(0, 500))
            })

    df = pd.DataFrame(data)
    df['log_mortality'] = np.log(df['deaths'] / df['exposure'])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Grid search (same logic)
    ages = sorted(df['age'].unique())
    years = sorted(df['year'].unique())
    n_ages = len(ages)
    n_years = len(years)

    log_mort_matrix = df.pivot(index='age', columns='year', values='log_mortality').values
    log_mort_matrix = np.nan_to_num(log_mort_matrix, nan=0.0)

    param1_range = np.linspace(0.01, 0.05, 10)
    param2_range = np.linspace(0.0005, 0.005, 10)

    best_score = float('inf')
    best_params = None
    optimization_history = []

    for param1 in param1_range:
        for param2 in param2_range:
            predicted = calculate_reasonable_prediction(param1, param2, ages, years)
            score = np.mean((predicted - log_mort_matrix)**2)

            optimization_history.append({
                'iteration': len(optimization_history) + 1,
                'param1': param1,
                'param2': param2,
                'score': score
            })

            if score < best_score:
                best_score = score
                best_params = {
                    'param1': param1,
                    'param2': param2,
                    'score': score,
                    'predicted': predicted
                }

    # Create comprehensive visualizations
    create_convergence_dashboard(optimization_history, best_params, ages, years, log_mort_matrix)

    return best_params, optimization_history

def calculate_reasonable_prediction(param1, param2, ages, years):
    """Calculate Lee-Carter style predictions"""
    n_ages = len(ages)
    n_years = len(years)

    a_x = -4.0 + param1 * np.array(ages)
    b_x = param2 * np.exp(-np.array(ages) / 25)
    k_t = np.linspace(-1, 1, n_years)

    predicted = np.zeros((n_ages, n_years))
    for i in range(n_ages):
        for j in range(n_years):
            predicted[i, j] = a_x[i] + b_x[i] * k_t[j]

    return predicted

def create_convergence_dashboard(history, best_params, ages, years, actual_matrix):
    """Create comprehensive convergence visualization dashboard"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    fig.suptitle('Lee-Carter Model: Grid Search Convergence Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. MSE Convergence (linear)
    ax1 = fig.add_subplot(gs[0, 0])
    iterations = [h['iteration'] for h in history]
    scores = [h['score'] for h in history]

    ax1.plot(iterations, scores, 'b-', alpha=0.7, linewidth=1.5, label='MSE')
    ax1.axhline(y=best_params['score'], color='r', linestyle='--',
                label='Best: {:.2f}'.format(best_params['score']))
    ax1.set_title('MSE Convergence\n(Linear Scale)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Squared Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MSE Convergence (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(iterations, scores, 'g-', linewidth=2, label='MSE')
    ax2.axhline(y=best_params['score'], color='r', linestyle='--')
    ax2.set_title('MSE Convergence\n(Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE (log scale)')
    ax2.grid(True, alpha=0.3)

    # 3. Running Best MSE
    ax3 = fig.add_subplot(gs[0, 2])
    running_min = np.minimum.accumulate(scores)
    ax3.plot(iterations, running_min, 'purple', linewidth=2.5, label='Running Best')
    ax3.set_title('Running Best MSE', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Best MSE So Far')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Convergence Rate (Improvement)
    ax4 = fig.add_subplot(gs[0, 3])
    improvement = [abs(scores[i] - running_min[i]) for i in range(len(scores))]
    ax4.semilogy(iterations, improvement, 'orange', linewidth=1.5, label='Improvement')
    ax4.set_title('Convergence Improvement\n(Distance from Best)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('|MSE - Best| (log scale)')
    ax4.grid(True, alpha=0.3)

    # 5. Parameter Space - Param1 vs MSE
    ax5 = fig.add_subplot(gs[1, 0])
    param1_values = [h['param1'] for h in history]
    scatter5 = ax5.scatter(param1_values, scores, c=iterations, cmap='viridis',
                          s=50, alpha=0.6)
    ax5.plot(best_params['param1'], best_params['score'], 'r*', markersize=15, label='Best')
    ax5.set_title('Parameter Space\n(Param1 vs MSE)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Param1 (Age Effect Scale)')
    ax5.set_ylabel('MSE')
    ax5.legend()
    plt.colorbar(scatter5, ax=ax5, label='Iteration')

    # 6. Parameter Space - Param2 vs MSE
    ax6 = fig.add_subplot(gs[1, 1])
    param2_values = [h['param2'] for h in history]
    scatter6 = ax6.scatter(param2_values, scores, c=iterations, cmap='viridis',
                          s=50, alpha=0.6)
    ax6.plot(best_params['param2'], best_params['score'], 'r*', markersize=15, label='Best')
    ax6.set_title('Parameter Space\n(Param2 vs MSE)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Param2 (Age Slope Scale)')
    ax6.set_ylabel('MSE')
    ax6.legend()
    plt.colorbar(scatter6, ax=ax6, label='Iteration')

    # 7. 2D Parameter Space Heatmap
    ax7 = fig.add_subplot(gs[1, 2])
    param1_unique = sorted(set(param1_values))
    param2_unique = sorted(set(param2_values))

    # Create a meshgrid for the heatmap
    P1, P2 = np.meshgrid(param1_unique, param2_unique)
    scores_grid = np.zeros_like(P1)

    for i, p1 in enumerate(param1_unique):
        for j, p2 in enumerate(param2_unique):
            for h in history:
                if abs(h['param1'] - p1) < 1e-10 and abs(h['param2'] - p2) < 1e-10:
                    scores_grid[j, i] = h['score']
                    break

    im = ax7.imshow(scores_grid, extent=[min(param1_unique), max(param1_unique),
                                        min(param2_unique), max(param2_unique)],
                    aspect='auto', origin='lower', cmap='YlOrRd', norm=LogNorm(vmin=0.001, vmax=1000))
    ax7.plot(best_params['param1'], best_params['param2'], 'b*', markersize=15, label='Best')
    ax7.set_title('Parameter Space\nHeatmap', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Param1')
    ax7.set_ylabel('Param2')
    plt.colorbar(im, ax=ax7, label='MSE')
    ax7.legend()

    # 8. Convergence Rate Analysis
    ax8 = fig.add_subplot(gs[1, 3])
    convergence_rates = []
    for i in range(1, len(scores)):
        rate = (scores[i-1] - scores[i]) / scores[i-1] if scores[i-1] > 0 else 0
        convergence_rates.append(rate)

    ax8.plot(range(1, len(scores)), convergence_rates, 'r-', linewidth=1, alpha=0.7)
    ax8.set_title('Convergence Rate\n(Per Iteration)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Improvement Rate')
    ax8.grid(True, alpha=0.3)

    # 9. Sample Predictions (Actual vs Predicted)
    ax9 = fig.add_subplot(gs[2, :2])
    age_indices = [20, 40, 60, 80]
    colors = ['blue', 'green', 'red', 'purple']

    for idx, age_idx in enumerate(age_indices):
        if age_idx < len(ages):
            actual_values = actual_matrix[age_idx, :]
            predicted_values = best_params['predicted'][age_idx, :]

            ax9.plot(years, actual_values, 'o-', color=colors[idx],
                    label='Age {} (Actual)'.format(ages[age_idx]), linewidth=2, alpha=0.8)
            ax9.plot(years, predicted_values, 's--', color=colors[idx],
                    label='Age {} (Predicted)'.format(ages[age_idx]), linewidth=2, alpha=0.6)

    ax9.set_title('Model Fit: Actual vs Predicted\n(Sample Ages)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Log Mortality Rate')
    ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax9.grid(True, alpha=0.3)

    # 10. Residual Analysis
    ax10 = fig.add_subplot(gs[2, 2])
    residuals = actual_matrix - best_params['predicted']
    ax10.hist(residuals.flatten(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    ax10.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Residual')
    ax10.set_ylabel('Frequency')
    ax10.axvline(x=0, color='red', linestyle='--')
    ax10.grid(True, alpha=0.3)

    # 11. Summary Statistics
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')

    # Calculate statistics
    actual_clean = actual_matrix.flatten()
    predicted_clean = best_params['predicted'].flatten()
    mask = ~np.isnan(actual_clean) & ~np.isnan(predicted_clean)

    ss_res = np.sum((actual_clean[mask] - predicted_clean[mask])**2)
    ss_tot = np.sum((actual_clean[mask] - np.mean(actual_clean[mask]))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(best_params['score'])

    summary_text = """
    OPTIMIZATION SUMMARY
    ====================

    Best Parameters:
      Param1: {:.6f}
      Param2: {:.6f}
      MSE:    {:.6f}

    Model Performance:
      R-squared: {:.4f}
      RMSE:      {:.6f}

    Convergence:
      Iterations:  {}
      Best Iter:   {}
      Status:      {}
    """.format(
        best_params['param1'], best_params['param2'], best_params['score'],
        r_squared, rmse,
        len(history), min(history, key=lambda x: x['score'])['iteration'],
        "Good" if best_params['score'] < 5 else "Needs Refinement"
    )

    ax11.text(0.05, 0.5, summary_text, fontsize=10, fontfamily='monospace',
              verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5",
              facecolor="lightgray", alpha=0.3))

    plt.tight_layout()
    plt.show()

    return fig

# Execute the visualization
if __name__ == "__main__":
    print("Creating comprehensive Lee-Carter convergence visualizations...")
    print("Note: Install matplotlib with: pip install matplotlib")

    try:
        results = create_comprehensive_convergence_demo()
        print("Visualization completed successfully!")
    except Exception as e:
        print("To run visualizations, install matplotlib: pip install matplotlib")
        print("Error: {}".format(e))