import pandas as pd
import numpy as np


# ================================================================================================================
# 1. CREATE TOY PATHWAY RELATIONS WITH LOOP  (A, B as inputs; C, D, E as internal/output) and a feedback loop D→C.
# ================================================================================================================
# Network structure:
#   A (input) ─→ C ─→ E (output)
#   B (input) ─→ C ─→ D ─→ E
#                ↑_________|  (LOOP: D feeds back to C)
#
# This creates a feedback loop: C→D→C
# The network structure:
#A → C (activation)
#B → C (activation)
#C → D (activation)
#D → C (activation) ← This creates the loop
#C → E (inhibition)
pathway_data = {
    'pathway': ['test_pathway'] * 5,
    'source': ['A', 'B', 'C', 'D', 'C'],
    'interactiontype': ['activation', 'activation', 'activation', 'activation', 'inhibition'],
    'target': ['C', 'C', 'D', 'C', 'E']
}
pathway_df = pd.DataFrame(pathway_data)
pathway_df.to_csv('./data/test_pathway_relations.csv', index=False)
print("Created: test_pathway_relations.csv")
print(pathway_df)
print()

# ============================================================================
# 2. CREATE SIMULATED UDP VALUES (3 samples, 5 genes: A, B, C, D, E)
# ============================================================================
# Rows = genes, Columns = samples
udp_data = {
    'sample1': [0.9, 0.8, 0.5, 0.5, 0.5],  # A, B high → expect high activity
    'sample2': [0.2, 0.1, 0.5, 0.5, 0.5],  # A, B low → expect low activity
    'sample3': [0.9, 0.1, 0.5, 0.5, 0.5]   # A high, B low → expect medium activity
}
udp_df = pd.DataFrame(udp_data, index=['a', 'b', 'c', 'd', 'e'])
udp_df.to_csv('./data/test_output_udp.csv')
print("Created: test_output_udp.csv")
print(udp_df)
print()

# ============================================================================
# 3. MANUAL CALCULATION OF GROUND TRUTH
# ============================================================================
# CPD parameters (matching activity.py defaults):
# - activation weight = 0.85
# - inhibition weight = 0.15
# - baseline = 0.1

def noisy_or(beliefs):
    """Noisy-OR combination of beliefs"""
    return 1.0 - np.prod([1.0 - b for b in beliefs])

def compute_ground_truth(udp_a, udp_b, max_iter=30, tol=1e-3):
    """Manually compute LBP convergence for ground truth"""
    cpd_activation = 0.85
    cpd_inhibition = 0.15
    cpd_baseline = 0.1
    
    # Initialize beliefs
    beliefs = {'a': udp_a, 'b': udp_b, 'c': 0.5, 'd': 0.5, 'e': 0.5}
    
    for iteration in range(max_iter):
        old_beliefs = beliefs.copy()
        
        # Update C: receives from A (activation), B (activation), D (activation - loop)
        # Messages:
        msg_a_to_c = cpd_baseline + cpd_activation * beliefs['a']
        msg_b_to_c = cpd_baseline + cpd_activation * beliefs['b']
        msg_d_to_c = cpd_baseline + cpd_activation * beliefs['d']
        beliefs['c'] = noisy_or([msg_a_to_c, msg_b_to_c, msg_d_to_c])
        
        # Update D: receives from C (activation)
        msg_c_to_d = cpd_baseline + cpd_activation * old_beliefs['c']
        beliefs['d'] = msg_c_to_d
        
        # Update E: receives from C (inhibition)
        # For inhibition: P(E=Up) = baseline * (1 - combined_input)
        msg_c_to_e = cpd_inhibition * (1.0 - old_beliefs['c'])
        beliefs['e'] = msg_c_to_e
        
        # Check convergence
        max_change = max(abs(beliefs[g] - old_beliefs[g]) for g in ['c', 'd', 'e'])
        if max_change < tol:
            break
    
    # Pathway activity = mean of all beliefs
    pathway_activity = np.mean(list(beliefs.values()))
    
    return pathway_activity, beliefs

# Calculate ground truth for each sample
results = []
for sample_name, (udp_a, udp_b) in [
    ('sample1', (0.9, 0.8)),
    ('sample2', (0.2, 0.1)),
    ('sample3', (0.9, 0.1))
]:
    activity, final_beliefs = compute_ground_truth(udp_a, udp_b)
    results.append({
        'sample': sample_name,
        'pathway_activity': activity,
        'belief_a': final_beliefs['a'],
        'belief_b': final_beliefs['b'],
        'belief_c': final_beliefs['c'],
        'belief_d': final_beliefs['d'],
        'belief_e': final_beliefs['e']
    })

result_df = pd.DataFrame(results)
result_df.to_csv('./data/test_expected_results.csv', index=False)
print("Created: test_expected_results.csv")
print(result_df)
print()

# ============================================================================
# 4. VERIFICATION INSTRUCTIONS
# ============================================================================
print("=" * 70)
print("QA TEST INSTRUCTIONS:")
print("=" * 70)
print("""
1. Use test_pathway_relations.csv as your pathway file
2. Use test_output_udp.csv as your UDP input
3. Run your LBP activity calculation
4. Compare your results to test_expected_results.csv

Expected behavior:
- Sample1 (high A, high B): High pathway activity (~0.6-0.7)
- Sample2 (low A, low B): Low pathway activity (~0.3-0.4)  
- Sample3 (high A, low B): Medium pathway activity (~0.5-0.6)

The loop (D→C) should converge within 10-15 iterations.
Beliefs should match expected results within tolerance of 0.01.
""")