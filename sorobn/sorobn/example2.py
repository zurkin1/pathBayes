import sorobn

# Example 1: Using the built-in alarm network example
bn = sorobn.examples.alarm()

# Sample from the network
samples = bn.sample(n=1000)
print(samples.head())
print(f"\nGenerated {len(samples)} samples")
print(f"Columns: {list(samples.columns)}")

# Query probabilities
prob = bn.query('Alarm', event={'Burglary': True, 'Earthquake': False})
print(f"\nP(Alarm | Burglary=True, Earthquake=False) = {prob}")

# Example 2: Creating your own simple Bayesian Network
# Let's create a simple "Sprinkler" network:
# Rain -> Grass Wet <- Sprinkler

bn_custom = sorobn.BayesNet()

# Add nodes with their conditional probability tables
bn_custom.add_node(
    'Rain',
    cdt={
        (): {True: 0.2, False: 0.8}  # P(Rain) - no parents
    }
)

bn_custom.add_node(
    'Sprinkler',
    cdt={
        (): {True: 0.1, False: 0.9}  # P(Sprinkler) - no parents
    }
)

bn_custom.add_node(
    'GrassWet',
    cdt={
        # P(GrassWet | Rain, Sprinkler)
        (True, True): {True: 0.99, False: 0.01},
        (True, False): {True: 0.90, False: 0.10},
        (False, True): {True: 0.85, False: 0.15},
        (False, False): {True: 0.05, False: 0.95},
    },
    parents=['Rain', 'Sprinkler']
)

# Sample from custom network
samples_custom = bn_custom.sample(n=500)
print("\n=== Custom Network Samples ===")
print(samples_custom.head(10))

# Query the custom network
prob_wet = bn_custom.query('GrassWet', event={'Rain': True})
print(f"\nP(GrassWet=True | Rain=True) = {prob_wet[True]:.3f}")