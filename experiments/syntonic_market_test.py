import math
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- EXCLUSIVE SYNTONIC LIBRARY IMPORTS ---
# We use your State object as the fundamental data structure
from syntonic import state as StateFactory
from syntonic.core.state import State
from syntonic.srt.constants import PHI_NUMERIC as PHI
from syntonic.crt.operators.syntony import syntony_spectral
from syntonic.srt.spectral.heat_kernel import HeatKernel # For theoretical comparisons

print(f"Syntonic Kernel Initialized | φ: {PHI:.6f}")

# =============================================================================
# 1. PURE PYTHON MLE ESTIMATOR (For Empirical Gamma)
# =============================================================================
def calculate_gamma_mle(data: List[float], tail_percentile: float = 0.20) -> float:
    """
    Calculates the Power Law tail exponent (γ) using Maximum Likelihood Estimation.
    Formula: γ = 1 + n / Σ ln(x_i / x_min)
    
    This replaces the 'powerlaw' library with raw math.
    """
    # 1. Filter for absolute returns > 0
    abs_data = sorted([abs(x) for x in data if x != 0])
    n_total = len(abs_data)
    if n_total < 10: return 0.0
    
    # 2. Define tail cutoff
    cutoff_idx = int(n_total * (1 - tail_percentile))
    tail = abs_data[cutoff_idx:]
    n_tail = len(tail)
    
    if n_tail < 5: return 0.0
    
    x_min = tail[0]
    
    # 3. MLE Calculation
    # Sum of log ratios
    sum_log = sum(math.log(x / x_min) for x in tail)
    if sum_log == 0: return 0.0
    
    gamma = 1 + n_tail / sum_log
    return gamma

# =============================================================================
# 2. SYNTONIC ADAPTER
# =============================================================================
def fetch_and_transform(ticker: str, period="2y") -> State:
    """
    Fetches raw data and immediately transmutes it into a Syntonic State.
    No Pandas DataFrames are retained.
    """
    # Raw download
    df = yf.download(ticker, period=period, progress=False)
    
    # Extract closing prices to pure list
    try:
        # Handle multi-level column mess from yfinance
        if 'Adj Close' in df:
            series = df['Adj Close']
        elif 'Close' in df:
            series = df['Close']
        else:
            return None
            
        # If series is a DataFrame (multi-ticker), extract the specific column
        if hasattr(series, 'columns') and ticker in series.columns:
            prices = series[ticker].dropna().tolist()
        else:
            prices = series.dropna().tolist()
    except Exception as e:
        print(f"Error converting {ticker}: {e}")
        return None
        
    # Compute Log Returns: ln(P_t / P_{t-1})
    # This represents the "Energy" of the price action
    log_returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0 and prices[i] > 0:
            r = math.log(prices[i] / prices[i-1])
            log_returns.append(r)
            
    # Create the State object
    # We map this to your Rust backend via the State wrapper
    psi = StateFactory(log_returns)
    return psi

# =============================================================================
# 3. THE SYNTONIC TEST SUITE
# =============================================================================
def run_syntonic_analysis():
    # A spectrum of assets from "High Utility" to "High Radioactive Decay"
    tickers = ["^TNX", "MSFT", "TSLA", "GME", "AMC", "BTC-USD", "ETH-USD"]
    
    results = []
    
    print("\n" + "="*110)
    print(f"{'ASSET':<10} | {'S (Ψ)':<8} | {'γ (Pred)':<8} | {'γ (Act)':<8} | {'Error%':<7} | {'Vol':<6} | {'Physics Class'}")
    print("-" * 110)
    
    for ticker in tickers:
        # 1. Ingest Data as State
        psi = fetch_and_transform(ticker)
        if psi is None or psi.size < 50:
            continue
            
        # 2. Pre-process State using State methods
        # Tanh normalization dampens outliers to reveal the spectral structure
        psi_normalized = psi.tanh()
        
        # 3. Calculate Syntony (S)
        # Using syntony_spectral from your operators library
        # This measures Low-Frequency Concentration vs High-Frequency Entropy
        s_val = syntony_spectral(psi_normalized)
        
        # 4. The Prediction (SRT Postulate)
        # γ = 1 + S
        gamma_pred = 1 + s_val
        
        # 5. The Empirical Measurement
        # We extract the raw list from the State to run MLE
        raw_data = psi.to_list()
        gamma_act = calculate_gamma_mle(raw_data)
        
        # 6. Volatility (Social Heat)
        # Standard deviation of the State
        vol = math.sqrt(sum((x - sum(raw_data)/len(raw_data))**2 for x in raw_data) / (len(raw_data)-1)) * math.sqrt(252)
        
        # 7. Classification
        if s_val > 0.618: # Golden Ratio
            p_class = "Stable (Harmonic)"
        elif s_val > 0.3:
            p_class = "Dynamic"
        else:
            p_class = "Radioactive (Meme)"
            
        error = abs((gamma_act - gamma_pred) / gamma_pred) * 100
        
        print(f"{ticker:<10} | {s_val:<8.4f} | {gamma_pred:<8.4f} | {gamma_act:<8.4f} | {error:<7.2f} | {vol:<6.2f} | {p_class}")
        
        results.append({
            'ticker': ticker,
            's': s_val,
            'gamma': gamma_act,
            'vol': vol
        })

    print("="*110)
    return results

def visualize_physics(results):
    """
    Visualizes the results to confirm the Postulate.
    """
    s_vals = [r['s'] for r in results]
    g_vals = [r['gamma'] for r in results]
    v_vals = [r['vol'] for r in results]
    labels = [r['ticker'] for r in results]
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: The Syntonic Law (γ = 1 + S)
    ax1.scatter(s_vals, g_vals, c=s_vals, cmap='coolwarm', s=150)
    
    # Theoretical Limit
    x_line = [i/100 for i in range(101)]
    y_line = [1 + x for x in x_line]
    ax1.plot(x_line, y_line, color='#00ff00', linestyle='--', label='Theory: γ = 1 + S')
    
    ax1.set_title(f"Quantized Faith: Syntony vs Tail Exponent")
    ax1.set_xlabel("Syntony Index S(Ψ)")
    ax1.set_ylabel("Stability Parameter γ")
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    for i, txt in enumerate(labels):
        ax1.annotate(txt, (s_vals[i], g_vals[i]), xytext=(5,5), textcoords='offset points', color='white')

    # Plot 2: The Social Heat Kernel
    # Volatility vs (1-S)
    faith_inv = [math.log(1 - s + 1e-6) for s in s_vals]
    log_vol = [math.log(v) for v in v_vals]
    
    ax2.scatter(faith_inv, log_vol, color='magenta', s=100)
    ax2.set_title("Social Heat Kernel: Volatility Scaling")
    ax2.set_xlabel("log(1 - S) [Instability]")
    ax2.set_ylabel("log(Volatility)")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = run_syntonic_analysis()
    visualize_physics(data)