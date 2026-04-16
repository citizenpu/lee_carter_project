import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_clean_mortality_data():
    """Create clean, well-behaved mortality data for demonstration"""
    np.random.seed(42)

    years = np.arange(2000, 2020)
    ages = np.arange(0, 100)

    # Create Lee-Carter structure with proper scaling
    data = []
    for year in years:
        for age in ages:
            # Lee-Carter model: log(mortality) = a_x + b_x * k_t
            a_x = -4.0 + 0.02 * age - 0.0001 * age**2  # Age effects (reasonable range)
            b_x = 0.001 * np.exp(-age / 25)  # Age slopes (decreasing with age)
            k_t = -0.5 + 0.05 * (year - 2000)  # Period effects (improving over time)

            log_mortality = a_x + b_x * k_t

            # Ensure reasonable mortality rates (0.001% to 10%)
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

    # Remove any NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

def simple_grid_search_optimization():
    """Simplified grid search that will converge properly"""

    # Create clean synthetic data
    df = create_clean_mortality_data()

    print("Data dimensions: {} rows x {} columns".format(df.shape[0], df.shape[1]))
    print("Age range: {} to {}".format(df['age'].min(), df['age'].max()))
    print("Year range: {} to {}".format(df['year'].min(), df['year'].max()))

    # Prepare data
    ages = sorted(df['age'].unique())
    years = sorted(df['year'].unique())
    n_ages = len(ages)
    n_years = len(years)

    # Create log mortality matrix
    log_mort_matrix = df.pivot(index='age', columns='year', values='log_mortality').values

    # Remove any remaining NaN values
    log_mort_matrix = np.nan_to_num(log_mort_matrix, nan=0.0)

    # Grid search setup
    param1_range = np.linspace(0.01, 0.05, 10)  # Age effect scale
    param2_range = np.linspace(0.0005, 0.005, 10)  # Age slope scale

    best_score = float('inf')
    best_params = None
    optimization_history = []

    total_iterations = len(param1_range) * len(param2_range)
    print("\nStarting grid search: {} parameter combinations".format(total_iterations))

    # Grid search loop
    iteration = 0
    for param1 in param1_range:
        for param2 in param2_range:
            iteration += 1

            try:
                # Calculate predictions using Lee-Carter structure
                predicted = calculate_reasonable_prediction(param1, param2, ages, years)

                # Calculate score (MSE)
                score = np.mean((predicted - log_mort_matrix)**2)

                # Track optimization history
                optimization_history.append({
                    'iteration': iteration,
                    'param1': param1,
                    'param2': param2,
                    'score': score
                })

                # Update best parameters
                if score < best_score:
                    best_score = score
                    best_params = {
                        'param1': param1,
                        'param2': param2,
                        'score': score,
                        'predicted': predicted
                    }

                    print("  Iteration {}/{}: NEW BEST".format(iteration, total_iterations))
                    print("     Param1: {:.6f}, Param2: {:.6f}".format(param1, param2))
                    print("     Score (MSE): {:.8f}".format(score))
                else:
                    # Progress indicator
                    if iteration % 15 == 0:
                        print("  Progress {}/{}: Best MSE = {:.8f}".format(iteration, total_iterations, best_score))

            except Exception as e:
                print("  Warning: Iteration {} failed: {}".format(iteration, e))
                continue

    print("\nGrid search completed in {} iterations".format(iteration))
    return {
        'best_params': best_params,
        'optimization_history': optimization_history,
        'ages': ages,
        'years': years,
        'actual': log_mort_matrix
    }

def calculate_reasonable_prediction(param1, param2, ages, years):
    """Calculate Lee-Carter style predictions with proper scaling"""
    n_ages = len(ages)
    n_years = len(years)

    # Create properly scaled parameters
    a_x = -4.0 + param1 * np.array(ages)  # Age effects
    b_x = param2 * np.exp(-np.array(ages) / 25)  # Age slopes
    k_t = np.linspace(-1, 1, n_years)  # Period effects

    # Calculate predictions
    predicted = np.zeros((n_ages, n_years))
    for i in range(n_ages):
        for j in range(n_years):
            predicted[i, j] = a_x[i] + b_x[i] * k_t[j]

    return predicted

def show_convergence_results(results):
    """Show detailed convergence results"""

    if results is None or results['best_params'] is None:
        print("Error: No valid results to display")
        return

    best_params = results['best_params']
    history = results['optimization_history']

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)

    print("\nData Information:")
    print("   Age range: {} to {}".format(min(results['ages']), max(results['ages'])))
    print("   Year range: {} to {}".format(min(results['years']), max(results['years'])))
    print("   Total observations: {}".format(results['actual'].size))

    print("\nOptimization Process:")
    print("   Total iterations: {}".format(len(history)))

    if len(history) > 0:
        best_iter = min(history, key=lambda x: x['score'])['iteration']
        print("   Best iteration: {}".format(best_iter))
        print("   Final MSE: {:.8f}".format(best_params['score']))
        print("   Param1: {:.6f}".format(best_params['param1']))
        print("   Param2: {:.6f}".format(best_params['param2']))

    # Calculate R-squared
    actual_clean = results['actual'].flatten()
    predicted_clean = best_params['predicted'].flatten()

    # Remove any remaining NaN values
    mask = ~np.isnan(actual_clean) & ~np.isnan(predicted_clean) & (actual_clean != 0)

    if np.sum(mask) > 0:
        ss_res = np.sum((actual_clean[mask] - predicted_clean[mask])**2)
        ss_tot = np.sum((actual_clean[mask] - np.mean(actual_clean[mask]))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print("\nModel Performance:")
        print("   R-squared: {:.4f}".format(r_squared))
        print("   RMSE: {:.6f}".format(np.sqrt(best_params['score'])))

        # Show sample predictions vs actual
        print("\nSample Predictions (Age 40):")
        age_40_idx = results['ages'].index(40) if 40 in results['ages'] else len(results['ages']) // 2
        for j, year in enumerate(results['years'][:5]):  # First 5 years
            actual_val = results['actual'][age_40_idx, j]
            pred_val = best_params['predicted'][age_40_idx, j]
            print("   Year {}: Actual = {:.6f}, Predicted = {:.6f}, Error = {:.6f}".format(
                year, actual_val, pred_val, abs(actual_val - pred_val)))

    # Convergence statistics
    if len(history) > 10:
        first_10_scores = [h['score'] for h in history[:10]]
        last_10_scores = [h['score'] for h in history[-10:]]

        print("\nConvergence Statistics:")
        print("   Average score (first 10): {:.8f}".format(np.mean(first_10_scores)))
        print("   Average score (last 10): {:.8f}".format(np.mean(last_10_scores)))
        print("   Improvement: {:.2f}%".format(((np.mean(first_10_scores) - best_params['score']) / np.mean(first_10_scores) * 100)))

        # Show convergence pattern
        print("\nConvergence Pattern:")
        for i in range(0, len(history), max(1, len(history)//5)):
            print("   Iteration {}: MSE = {:.8f}".format(history[i]['iteration'], history[i]['score']))

        # Analyze convergence
        first_half = [h['score'] for h in history[:len(history)//2]]
        second_half = [h['score'] for h in history[len(history)//2:]]

        print("\nConvergence Analysis:")
        print("   First half average MSE:  {:.8f}".format(np.mean(first_half)))
        print("   Second half average MSE: {:.8f}".format(np.mean(second_half)))
        print("   Overall improvement: {:.2f}%".format(((np.mean(first_half) - np.mean(second_half)) / np.mean(first_half) * 100)))

        # Convergence assessment
        if best_params['score'] < 0.01:
            print("   Status: Strong convergence achieved")
        elif best_params['score'] < 0.1:
            print("   Status: Good convergence")
        else:
            print("   Status: Moderate convergence - consider refining grid")

# Main execution
if __name__ == "__main__":
    print("Lee-Carter Mortality Model - Grid Search Convergence Demo")
    print("="*60)

    results = simple_grid_search_optimization()
    show_convergence_results(results)

    if results is not None and results['best_params'] is not None:
        print("\nDemo completed successfully!")
    else:
        print("\nDemo failed - check error messages above")