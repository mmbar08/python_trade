import math
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type="call", dividends=0):
    """
    Calculate the Black-Scholes price of a European option.

    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate (as decimal, e.g., 0.05)
    sigma : float : Implied volatility (as decimal, e.g., 0.25)
    option_type : str : "call" or "put"
    dividends : float : Present value of expected dividends (default 0)

    Returns:
    float : Option price
    """
    S_adj = S - dividends  # adjust stock price for dividends

    d1 = (math.log(S_adj / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == "call":
        price = S_adj * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price


# --- Example Usage ---
# Inputs
S = float(input("Current stock price (S): "))
K = float(input("Strike price (K): "))
T = float(input("Time to expiration in years (T): "))
r = float(input("Risk-free rate as decimal (r): "))
sigma = float(input("Implied volatility as decimal (sigma): "))
option_type = input("Option type (call/put): ")
dividends = float(input("Present value of dividends (if none, enter 0): "))

# Output
price = black_scholes_price(S, K, T, r, sigma, option_type, dividends)
print(f"Projected {option_type} option price: ${price:.2f}")
