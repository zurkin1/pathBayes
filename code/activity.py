import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from config import *


def is_inhibitory(interaction_type):
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type for keyword in inhibitory_keywords)


def get_pathway_genes(interactions):
    """Extract all unique genes from pathway interactions"""
    genes = set()
    for sources, _, targets in interactions:
        genes.update(sources)
        genes.update(targets)
    return genes


def get_gene_parents(gene, interactions):
    """Get all source genes that have this gene as target"""
    parents = []
    for sources, inttype, targets in interactions:
        if gene in targets:
            parents.append((sources, inttype))
    return parents


def compute_message(sources, inttype, beliefs):
    """
    Compute message from source genes through an interaction.
    
    Args:
        sources: list of source gene names
        inttype: interaction type string
        beliefs: dict of current belief values {gene: probability_up}
    
    Returns:
        message: probability value representing combined input
    """
    # Collect beliefs of source genes
    source_beliefs = [beliefs.get(gene, 0.5) for gene in sources]
    
    # Combine multiple inputs using noisy-OR for activation. P(output=Up) = 1 - prod(1 - P(source_i=Up)). Noisy-OR is a probabilistic model for combining multiple binary
    # causes leading to an effect. If any parent is "Up", the child tends to be "Up" with high probability. Formula: P(child=Up) = 1 - ∏(1 - P(parent_i=Up)).
    # It's computationally efficient and biologically realistic - multiple promoters can independently activate a target. Alternative is weighted sum, but noisy-OR better
    # captures "at least one input succeeds" logic common in signaling pathways.
    combined = 1.0 - np.prod([1.0 - b for b in source_beliefs])
    
    # Apply CPT weight based on interaction type
    if is_inhibitory(inttype):
        # For inhibition: high input → low output
        # P(output=Up | inhibited) = baseline * (1 - combined)
        message = CPT_INHIBITION * (1.0 - combined)
    else:
        # For activation: high input → high output
        # P(output=Up | activated) = baseline + weight * combined
        message = CPT_BASELINE + CPT_ACTIVATION * combined
    
    return np.clip(message, 0.0, 1.0)


def loopy_belief_propagation(interactions, initial_beliefs, max_iterations=30, tolerance=1e-3):
    """
    Run Loopy Belief Propagation on pathway network.
    
    Args:
        interactions: list of (sources, inttype, targets) tuples
        initial_beliefs: dict of {gene: udp_value} for input genes
        max_iterations: maximum number of LBP iterations
        tolerance: convergence threshold
    
    Returns:
        beliefs: dict of final belief values {gene: probability_up}
    """
    # Get all genes in pathway
    all_genes = get_pathway_genes(interactions)
    
    # Initialize beliefs
    beliefs = {gene: initial_beliefs.get(gene, 0.5) for gene in all_genes}
    
    # Identify input genes (genes with no parents in the pathway)
    input_genes = set()
    target_genes = set()
    for sources, _, targets in interactions:
        target_genes.update(targets)
    input_genes = all_genes - target_genes
    
    # LBP iteration
    for iteration in range(max_iterations):
        old_beliefs = beliefs.copy()
        new_beliefs = {}
        
        # Update beliefs for all non-input genes
        for gene in all_genes:
            if gene in input_genes:
                # Input genes keep their initial UDP values
                new_beliefs[gene] = initial_beliefs.get(gene, 0.5)
            else:
                # Get all incoming messages from parent interactions
                parents = get_gene_parents(gene, interactions)
                
                if not parents:
                    # No parents, keep previous belief
                    new_beliefs[gene] = old_beliefs[gene]
                else:
                    # Compute messages from all parent interactions
                    messages = []
                    for sources, inttype in parents:
                        msg = compute_message(sources, inttype, old_beliefs)
                        messages.append(msg)
                    
                    # Combine messages using noisy-OR
                    if len(messages) == 1:
                        propagated_belief = messages[0]
                    else:
                        # Multiple parents: noisy-OR combination
                        combined = 1.0 - np.prod([1.0 - m for m in messages])
                        propagated_belief = combined
    
                    # Combine propagated belief with original UDP value.
                    original_udp = initial_beliefs.get(gene, 0.5)
                    new_beliefs[gene] = (UDP_WEIGHT * original_udp + 
                                        (1 - UDP_WEIGHT) * propagated_belief)
        
        beliefs = new_beliefs
        
        # Check convergence
        max_change = max(abs(beliefs[g] - old_beliefs[g]) for g in all_genes)
        if max_change < tolerance:
            break
    
    return beliefs


def process_sample(args):
    """Process all pathways for a single sample with LBP"""
    sample_idx, sample_udp, pathway_interactions = args
    pathway_activities = {}
    pathway_beliefs = {}
    
    for pathway, interactions in pathway_interactions.items():
        # Get UDP values for this sample
        pathway_genes = get_pathway_genes(interactions)
        initial_beliefs = {gene: sample_udp.get(gene, 0.5) for gene in pathway_genes}
        
        # Run LBP
        final_beliefs = loopy_belief_propagation(interactions, initial_beliefs)
        
        # Calculate pathway activity as mean of all gene beliefs
        pathway_activity = np.mean(list(final_beliefs.values()))
        
        pathway_activities[pathway] = pathway_activity
        pathway_beliefs[pathway] = final_beliefs
    
    return sample_idx, pathway_activities, pathway_beliefs


def calc_activity(udp_file=f'./data/{TEST}output_udp.csv'):
    """
    Calculate pathway activities using Bayesian UDP propagation.
    
    Args:
        udp_file: path to UDP values CSV
    """
    # Load UDP values
    """Load precomputed UDP values from udp.py output"""
    udp_df = pd.read_csv(udp_file, index_col=0)
    udp_df.index = udp_df.index.str.lower()
   
    # Load pathway relations
    pathway_relations = pd.read_csv(f'./data/{TEST}pathway_relations.csv')
    pathway_relations['source'] = pathway_relations['source'].fillna('').astype(str).str.lower().str.split('*')
    pathway_relations['target'] = pathway_relations['target'].fillna('').astype(str).str.lower().str.split('*')
    
    # Parse pathway interactions
    pathway_interactions = {}
    for _, row in pathway_relations.iterrows():
        pathway = row['pathway']
        sources = row['source']
        targets = row['target']
        inttype = row['interactiontype']
        if pathway not in pathway_interactions:
            pathway_interactions[pathway] = []
        pathway_interactions[pathway].append((sources, inttype, targets))
       
    # Initialize storage
    pathway_activities = {pathway: [] for pathway in pathway_interactions.keys()}
    n_samples = len(udp_df.columns)
    ordered_results = [None] * n_samples
    
    # Process samples in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        sample_args = []
        for idx in range(n_samples):
            sample_name = udp_df.columns[idx]
            # Get UDP values for this sample
            if sample_name in udp_df.columns:
                sample_udp = udp_df[sample_name].to_dict()
            else:
                print(f"Warning: Sample {sample_name} not found in UDP file")
                sample_udp = {}
            
            sample_args.append((idx, sample_udp, pathway_interactions))
        
        futures = [executor.submit(process_sample, arg) for arg in sample_args]
        
        for future in as_completed(futures):
            idx, sample_pathway_activities, sample_pathway_beliefs = future.result()
            ordered_results[idx] = sample_pathway_activities
            print(f"Processed sample {idx+1}/{n_samples}", end='\r')
    
    # Collect results in order
    for idx, sample_pathway_activities in enumerate(ordered_results):
        for pathway, activity in sample_pathway_activities.items():
            pathway_activities[pathway].append(activity)
    
    # Create activity matrix
    mean_activity_matrix = np.zeros((n_samples, len(pathway_interactions)))
    for idx, (pathway_name, activities) in enumerate(pathway_activities.items()):
        if activities:
            mean_activity_matrix[:, idx] = activities
        else:
            print(f"No activities for pathway {pathway_name}")
    
    # Create DataFrame
    activity_df = pd.DataFrame(
        mean_activity_matrix, 
        index=udp_df.columns, 
        columns=list(pathway_interactions.keys())
    ).T
    
    # Save results
    activity_df = activity_df.round(3)
    activity_df.T.to_csv(f'./data/{TEST}output_activity.csv')
    print(f"\nSaved activity matrix to ./data/{TEST}output_activity.csv")
    
    return activity_df


if __name__ == '__main__':
    calc_activity()