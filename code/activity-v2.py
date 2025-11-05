import pandas as pd
import pandas as pd
import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.inference import BeliefPropagation
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from config import *


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 
                          'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def create_interaction_factor(sources, target, interaction_type, interaction_id):
    """
    Create a factor representing an interaction's CPT.
    
    Args:
        sources: list of source gene names
        target: target gene name
        interaction_type: type of interaction
        interaction_id: unique identifier for this interaction
        
    Returns:
        Custom factor function
    """
    is_inhib = is_inhibitory(interaction_type)
    
    def factor_function(source_beliefs):
        """
        Compute interaction output given source beliefs.
        
        Args:
            source_beliefs: dict {gene_name: belief_value}
        
        Returns:
            probability value for target
        """
        # Get beliefs for source genes
        beliefs = [source_beliefs.get(gene, 0.5) for gene in sources]
        
        # Combine sources using noisy-OR (can be customized per interaction)
        combined = 1.0 - np.prod([1.0 - b for b in beliefs])
        
        # Apply CPT based on interaction type
        if is_inhib:
            # Inhibition: high input → low output
            output = CPT_BASELINE * (1.0 - CPT_INHIBITION * combined)
        else:
            # Activation: high input → high output  
            output = CPT_BASELINE + CPT_ACTIVATION * combined
        
        return np.clip(output, 0.0, 1.0)
    
    return factor_function


def build_factor_graph(interactions):
    """
    Build a factor graph from pathway interactions.
    
    Args:
        interactions: list of (sources, interaction_type, targets, pathway) tuples
        
    Returns:
        graph structure as dict and list of factor functions
    """
    # Collect all genes
    all_genes = set()
    for sources, _, targets, _ in interactions:
        all_genes.update(sources)
        all_genes.update(targets)
    
    # Create graph structure
    # gene_to_factors: which factors (interactions) affect each gene
    # factor_to_genes: which genes each factor (interaction) depends on
    gene_to_factors = {gene: [] for gene in all_genes}
    factor_to_genes = {}
    factor_functions = {}
    
    for idx, (sources, inttype, targets, pathway) in enumerate(interactions):
        interaction_id = f"interaction_{idx}"
        
        # Each interaction can have multiple targets
        for target in targets:
            # Create unique factor for each source->target connection
            factor_id = f"{interaction_id}_{target}"
            
            # Store which genes this factor depends on (sources + target)
            factor_to_genes[factor_id] = {
                'sources': sources,
                'target': target
            }
            
            # Store that this factor sends message to target
            gene_to_factors[target].append(factor_id)
            
            # Create factor function
            factor_functions[factor_id] = create_interaction_factor(
                sources, target, inttype, factor_id
            )
    
    return all_genes, gene_to_factors, factor_to_genes, factor_functions


def loopy_belief_propagation_manual(all_genes, gene_to_factors, factor_to_genes, 
                                    factor_functions, initial_beliefs, 
                                    max_iterations=30, tolerance=1e-3):
    """
    Manual belief propagation treating interactions as explicit factors.
    
    Args:
        all_genes: set of all gene names
        gene_to_factors: dict {gene: [factor_ids affecting it]}
        factor_to_genes: dict {factor_id: {'sources': [...], 'target': gene}}
        factor_functions: dict {factor_id: function}
        initial_beliefs: dict {gene: udp_value}
        max_iterations: maximum iterations
        tolerance: convergence threshold
        
    Returns:
        final beliefs dict {gene: probability}
    """
    # Initialize beliefs
    beliefs = {gene: initial_beliefs.get(gene, 0.5) for gene in all_genes}
    
    # Identify input genes (genes with no incoming factors)
    target_genes = set()
    for factor_info in factor_to_genes.values():
        target_genes.add(factor_info['target'])
    input_genes = all_genes - target_genes
    
    # Initialize messages
    # gene_to_factor_msgs[gene][factor] = message from gene to factor
    # factor_to_gene_msgs[factor][gene] = message from factor to gene
    gene_to_factor_msgs = {gene: {} for gene in all_genes}
    factor_to_gene_msgs = {factor: {} for factor in factor_to_genes}
    
    # Initialize all messages
    for gene in all_genes:
        for factor in gene_to_factors[gene]:
            factor_to_gene_msgs[factor][gene] = 0.5
    
    for factor_id, info in factor_to_genes.items():
        for source in info['sources']:
            gene_to_factor_msgs[source][factor_id] = beliefs[source]
    
    # Belief propagation iterations
    for iteration in range(max_iterations):
        old_beliefs = beliefs.copy()
        
        # PHASE 1: Gene to Factor messages
        for gene in all_genes:
            if gene in input_genes:
                # Input genes always send their UDP value
                for factor in gene_to_factors[gene]:
                    gene_to_factor_msgs[gene][factor] = initial_beliefs.get(gene, 0.5)
            else:
                # Non-input genes send belief excluding each factor
                for factor in gene_to_factors[gene]:
                    # Collect messages from all OTHER factors
                    other_msgs = [factor_to_gene_msgs[f][gene] 
                                 for f in gene_to_factors[gene] if f != factor]
                    
                    if other_msgs:
                        # Combine using product (in probability space)
                        msg = np.prod(other_msgs) * initial_beliefs.get(gene, 0.5)
                    else:
                        msg = initial_beliefs.get(gene, 0.5)
                    
                    gene_to_factor_msgs[gene][factor] = np.clip(msg, 0.0, 1.0)
        
        # PHASE 2: Factor to Gene messages  
        for factor_id, info in factor_to_genes.items():
            sources = info['sources']
            target = info['target']
            factor_func = factor_functions[factor_id]
            
            # Collect current beliefs of source genes
            source_beliefs = {s: gene_to_factor_msgs[s].get(factor_id, beliefs[s]) 
                             for s in sources}
            
            # Compute factor output
            output_msg = factor_func(source_beliefs)
            factor_to_gene_msgs[factor_id][target] = output_msg
        
        # PHASE 3: Update gene beliefs
        for gene in all_genes:
            if gene in input_genes:
                # Input genes keep their UDP value
                beliefs[gene] = initial_beliefs.get(gene, 0.5)
            else:
                # Combine all incoming factor messages
                incoming = [factor_to_gene_msgs[f][gene] 
                           for f in gene_to_factors[gene]]
                
                if incoming:
                    # Product of messages with prior
                    prior = initial_beliefs.get(gene, 0.5)
                    combined = np.prod(incoming) * prior
                    
                    # Normalize (simple approach)
                    beliefs[gene] = np.clip(combined, 0.0, 1.0)
                else:
                    beliefs[gene] = initial_beliefs.get(gene, 0.5)
        
        # Check convergence
        max_change = max(abs(beliefs[g] - old_beliefs[g]) for g in all_genes)
        if max_change < tolerance:
            break
    
    return beliefs


def process_sample(args):
    """Process a single sample with factor graph BP"""
    sample_idx, sample_udp, pathway_interactions = args
    pathway_activities = {}
    
    for pathway, interactions in pathway_interactions.items():
        # Build factor graph structure
        all_genes, gene_to_factors, factor_to_genes, factor_functions = \
            build_factor_graph(interactions)
        
        # Get UDP values for this pathway
        initial_beliefs = {gene: sample_udp.get(gene, 0.5) for gene in all_genes}
        
        # Run belief propagation
        final_beliefs = loopy_belief_propagation_manual(
            all_genes, gene_to_factors, factor_to_genes, factor_functions,
            initial_beliefs
        )
        
        # Calculate pathway activity as mean belief
        pathway_activity = np.mean(list(final_beliefs.values()))
        pathway_activities[pathway] = pathway_activity
    
    return sample_idx, pathway_activities


def calc_activity(udp_file=f'./data/{TEST}output_udp.csv'):
    """
    Calculate pathway activities using factor graph belief propagation.
    
    Args:
        udp_file: path to UDP values CSV
    """
    # Load UDP values
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
        
        # Store as (sources, inttype, targets, pathway)
        pathway_interactions[pathway].append((sources, inttype, targets, pathway))
    
    # Initialize storage
    n_samples = len(udp_df.columns)
    ordered_results = [None] * n_samples
    
    # Process samples in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        sample_args = []
        for idx in range(n_samples):
            sample_name = udp_df.columns[idx]
            sample_udp = udp_df[sample_name].to_dict() if sample_name in udp_df.columns else {}
            sample_args.append((idx, sample_udp, pathway_interactions))
        
        futures = [executor.submit(process_sample, arg) for arg in sample_args]
        
        for future in as_completed(futures):
            idx, sample_pathway_activities = future.result()
            ordered_results[idx] = sample_pathway_activities
            print(f"Processed sample {idx+1}/{n_samples}", end='\r')
    
    # Collect results
    pathway_activities = {pathway: [] for pathway in pathway_interactions.keys()}
    for sample_pathway_activities in ordered_results:
        for pathway, activity in sample_pathway_activities.items():
            pathway_activities[pathway].append(activity)
    
    # Create activity matrix
    activity_df = pd.DataFrame(
        pathway_activities,
        index=udp_df.columns
    ).T
    
    # Save results
    activity_df = activity_df.round(3)
    activity_df.T.to_csv(f'./data/{TEST}output_activity.csv')
    print(f"\nSaved activity matrix to ./data/{TEST}output_activity.csv")
    
    return activity_df


if __name__ == '__main__':
    calc_activity()