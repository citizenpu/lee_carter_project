import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_mortality_data():
    """Load mortality data and run Lee-Carter optimization without visualization"""

    try:
        # Try to load your actual data
        df = pd.read_excel("01全部数据资料-06死亡.xlsx")
        print(f"✓ Successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"⚠️ Could not load Excel file: {e}")
        print("Generating synthetic mortality data for demonstration...")
        df = create_synthetic_mortality_data()

    # Run grid search optimization
    print("\n" + "="*60)
    print("LEE-CARTER GRID SEARCH OPTIMIZATION")
    print("="*60)

    results = run_grid_search_optimization(df)

    if results is not None:
        # Print detailed results
        print_results_summary(results)
        return results
    else:
        print("❌ Optimization failed")
        return None

def create_synthetic_mortality_data():
    """Create realistic synthetic mortality data"""
    np.random.seed(42)

    years = np.arange(2000, 2020)
    ages = np.arange(0, 100)

    # Create Lee-Carter structure
    data = []
    for year in years:
        for age in ages:
            # Realistic mortality patterns
            base_mortality = 0.001 * np.exp(age / 20)  # Exponential increase with age
            time_trend = -0.02 * (year - 2000) / 20  # Improving over time
            age_trend = 0.0001 * age * (year - 2000) / 20  # Time effect varies by age

            mortality_rate = base_mortality * np.exp(time_trend + age_trend)
            mortality_rate += np.random.normal(0, 0.0001)  # Add noise

            data.append({
                'year': year,
                'age': age,
                'deaths': np.random.poisson(1000 * mortality_rate),
                'exposure': 10000 + np.random.normal(0, 500)
            })

    df = pd.DataFrame(data)
    df['log_mortality'] = np.log(df['deaths'] / df['exposure'])
    return df

def run_grid_search_optimization(df):
    """Perform grid search optimization for Lee-Carter parameters"""

    # Prepare data
    ages = sorted(df['age'].unique())
    years = sorted(df['year'].unique())
    n_ages = len(ages)
    n_years = len(years)

    print(f"📊 Data dimensions: {n_ages} ages × {n_years} years")
    print(f"📈 Age range: {min(ages)} to {max(ages)}")
    print(f"📅 Year range: {min(years)} to {max(years)}")

    # Create log mortality matrix
    log_mort_matrix = df.pivot(index='age', columns='year', values='log_mortality').values

    # Grid search setup
    tax_ranges = np.linspace(0.05, 0.45, 15)  # Tax-like parameters
    subsidy_ranges = np.linspace(0.001, 0.02, 15)  # Subsidy-like parameters

    best_score = np.inf
    best_params = None
    optimization_history = []

    total_iterations = len(tax_ranges) * len(subsidy_ranges)
    print(f"\n🔍 Starting grid search: {total_iterations} parameter combinations")

    # Grid search loop
    iteration = 0
    for tax_param in tax_ranges:
        for subsidy_param in subsidy_ranges:
            iteration += 1

            # Lee-Carter style calculation
            predicted = calculate_lee_carter_prediction(tax_param, subsidy_param, ages, years)

            # Calculate score (MSE)
            score = np.mean((predicted - log_mort_matrix)**2)

            # Track optimization history
            optimization_history.append({
                'iteration': iteration,
                'tax_param': tax_param,
                'subsidy_param': subsidy_param,
                'score': score
            })

            # Update best parameters
            if score < best_score:
                best_score = score
                best_params = {
                    'tax_param': tax_param,
                    'subsidy_param': subsidy_param,
                    'score': score,
                    'predicted': predicted
                }

                print(f"  📍 Iteration {iteration}/{total_iterations}: NEW BEST")
                print(f"     Tax param: {tax_param:.4f}, Subsidy param: {subsidy_param:.6f}")
                print(f"     Score (MSE): {score:.8f}")
            else:
                # Progress indicator
                if iteration % 20 == 0:
                    print(f"  ⏳ Iteration {iteration}/{total_iterations}: Best MSE = {best_score:.8f}")

    print(f"\n✅ Grid search completed in {iteration} iterations")
    return {
        'best_params': best_params,
        'optimization_history': optimization_history,
        'ages': ages,
        'years': years,
        'actual': log_mort_matrix
    }

def calculate_lee_carter_prediction(tax_param, subsidy_param, ages, years):
    """Calculate Lee-Carter style predictions"""
    n_ages = len(ages)
    n_years = len(years)

    # Lee-Carter structure: log_mortality = a_x + b_x * k_t
    a_x = -4.0 + tax_param * np.array(ages) / 100  # Age effects
    b_x = subsidy_param * np.exp(-np.array(ages) / 30)  # Age slopes
    k_t = np.linspace(0, 1, n_years)  # Period effects

    # Calculate predictions
    predicted = np.zeros((n_ages, n_years))
    for i in range(n_ages):
        for j in range(n_years):
            predicted[i, j] = a_x[i] + b_x[i] * k_t[j]

    return predicted

def print_results_summary(results):
    """Print detailed results summary"""

    if results is None:
        return

    best_params = results['best_params']
    history = results['optimization_history']

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)

    print(f"\n📊 Data Information:")
    print(f"   • Age range: {min(results['ages'])} to {max(results['ages'])}")
    print(f"   • Year range: {min(results['years'])} to {max(results['years'])}")
    print(f"   • Total observations: {results['actual'].size}")

    print(f"\n🔍 Optimization Process:")
    print(f"   • Total iterations: {len(history)}")
    print(f"   • Best iteration: {min(history, key=lambda x: x['score'])['iteration']}")
    print(f"   • Final MSE: {best_params['score']:.8f}")
    print(f"   • Tax parameter: {best_params['tax_param']:.6f}")
    print(f"   • Subsidy parameter: {best_params['subsidy_param']:.8f}")

    # Calculate R-squared
    actual_clean = results['actual'].flatten()
    predicted_clean = best_params['predicted'].flatten()
    mask = ~np.isnan(actual_clean) & ~np.isnan(predicted_clean)

    ss_res = np.sum((actual_clean[mask] - predicted_clean[mask])**2)
    ss_tot = np.sum((actual_clean[mask] - np.mean(actual_clean[mask]))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"\n📈 Model Performance:")
    print(f"   • R-squared: {r_squared:.4f}")
    print(f"   • RMSE: {np.sqrt(best_params['score']):.6f}")

    # Convergence statistics
    first_10_scores = [h['score'] for h in history[:10]]
    last_10_scores = [h['score'] for h in history[-10:]]

    print(f"\n⚡ Convergence Statistics:")
    print(f"   • Average score (first 10): {np.mean(first_10_scores):.8f}")
    print(f"   • Average score (last 10): {np.mean(last_10_scores):.8f}")
    print(f"   • Improvement: {((np.mean(first_10_scores) - best_params['score']) / np.mean(first_10_scores) * 100):.2f}%")

    # Show convergence pattern
    print(f"\n📈 Convergence Pattern (every 30 iterations):")
    for i in range(0, len(history), 30):
        print(f"   Iteration {history[i]['iteration']:3d}: MSE = {history[i]['score']:.8f}")

    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS:")
    print("="*60)

    # Analyze convergence
    first_half = [h['score'] for h in history[:len(history)//2]]
    second_half = [h['score'] for h in history[len(history)//2:]]

    print(f"First half average MSE:  {np.mean(first_half):.8f}")
    print(f"Second half average MSE: {np.mean(second_half):.8f}")
    print(f"Improvement: {((np.mean(first_half) - np.mean(second_half)) / np.mean(first_half) * 100):.2f}%")

    # Check if convergence is achieved
    if best_params['score'] < 0.0001:
        print("✅ Strong convergence achieved (MSE < 0.0001)")
    elif best_params['score'] < 0.001:
        print("✅ Good convergence achieved (MSE < 0.001)")
    else:
        print("⚠️  Moderate convergence - consider expanding parameter grid")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Lee-Carter Mortality Model Analysis")
    print("="*60)

    results = load_and_analyze_mortality_data()

    if results is not None:
        print("\n✅ Analysis completed successfully!")
        print("\nTo visualize the results, install matplotlib:")
        print("pip install matplotlib")
    else:
        print("\n❌ Analysis failed. Please check your data and parameters.")