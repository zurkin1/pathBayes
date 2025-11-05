import pandas as pd
import numpy as np
from config import *


def parse_pathway_interactions(relations_file):
    """Loads and parses pathway relations into a dict."""
    pathway_relations = pd.read_csv(relations_file)
    pathway_relations['source'] = pathway_relations['source'].fillna('').astype(str).str.lower().str.split('*')
    pathway_relations['target'] = pathway_relations['target'].fillna('').astype(str).str.lower().str.split('*')
    
    # Group interactions by pathway
    interactions_by_pathway = {}
    for _, row in pathway_relations.iterrows():
        pathway = row['pathway']
        if pathway not in interactions_by_pathway:
            interactions_by_pathway[pathway] = []
        
        interactions_by_pathway[pathway].append(
            (row['source'], row['interactiontype'], row['target'], pathway)
        )
    
    return interactions_by_pathway, list(interactions_by_pathway.keys())


# Load and parse interactions into simple data structures
pathway_interactions, pathway_names = parse_pathway_interactions(f'./data/{TEST}pathway_relations.csv')


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation',
                           'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def create_interaction_factor(sources, is_inhib):
    """
    Create a factor function representing an interaction.
    Pre-calculates inhibition status.
    """
    def factor_function(source_beliefs):
        """
        Compute interaction output given source beliefs.
        Args:
            source_beliefs: dict {gene_name: belief_value}
        Returns:
            probability value for target
        """
        beliefs = [source_beliefs.get(gene, 0.5) for gene in sources]
        
        # Combine sources using noisy-OR
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
    Build a factor graph structure from pathway interactions.
    """
    all_genes = set()
    gene_to_factors = {}
    factor_to_genes = {}
    factor_functions = {}
    
    for sources, _, targets, _ in interactions:
        all_genes.update(sources)
        all_genes.update(targets)

    for gene in all_genes:
        gene_to_factors[gene] = []

    for idx, (sources, inttype, targets, pathway) in enumerate(interactions):
        interaction_id = f"interaction_{idx}"
        is_inhib = is_inhibitory(inttype)
        
        for target in targets:
            factor_id = f"{interaction_id}_{target}"
            
            factor_to_genes[factor_id] = {'sources': sources, 'target': target}
            gene_to_factors[target].append(factor_id)
            factor_functions[factor_id] = create_interaction_factor(sources, is_inhib)
    
    return all_genes, gene_to_factors, factor_to_genes, factor_functions


def loopy_belief_propagation_manual(all_genes, gene_to_factors, factor_to_genes,
                                    factor_functions, initial_beliefs,
                                    max_iterations=30, tolerance=1e-3):
    """
    Manual belief propagation treating interactions as explicit factors.
    """
    beliefs = {gene: initial_beliefs.get(gene, 0.5) for gene in all_genes}
    
    target_genes = {info['target'] for info in factor_to_genes.values()}
    input_genes = all_genes - target_genes
    
    # Initialize messages
    factor_to_gene_msgs = {factor: {info['target']: 0.5} 
                           for factor, info in factor_to_genes.items()}
    gene_to_factor_msgs = {gene: {factor: beliefs[gene] 
                                  for factor in gene_to_factors.get(gene, [])} 
                           for gene in all_genes}
    
    for iteration in range(max_iterations):
        old_beliefs = beliefs.copy()
        
        # PHASE 1: Gene to Factor messages
        for gene, factors in gene_to_factors.items():
            if gene not in input_genes:
                for factor in factors:
                    other_msgs = [factor_to_gene_msgs[f][gene] 
                                  for f in factors if f != factor]
                    
                    msg = np.prod(other_msgs) if other_msgs else 1.0
                    msg *= initial_beliefs.get(gene, 0.5)
                    gene_to_factor_msgs[gene][factor] = np.clip(msg, 0.0, 1.0)
        
        # PHASE 2: Factor to Gene messages
        for factor_id, info in factor_to_genes.items():
            sources = info['sources']
            target = info['target']
            
            source_beliefs = {s: gene_to_factor_msgs[s].get(factor_id, beliefs[s]) 
                              for s in sources}
            
            factor_to_gene_msgs[factor_id][target] = factor_functions[factor_id](source_beliefs)
        
        # PHASE 3: Update gene beliefs
        for gene in all_genes:
            if gene not in input_genes:
                incoming = [factor_to_gene_msgs[f][gene] 
                            for f in gene_to_factors[gene]]
                
                if incoming:
                    combined = np.prod(incoming) * initial_beliefs.get(gene, 0.5)
                    beliefs[gene] = np.clip(combined, 0.0, 1.0)
            else:
                beliefs[gene] = initial_beliefs.get(gene, 0.5) # Input genes stay fixed
        
        # Check convergence
        if all_genes: # Avoid error on empty pathways
            max_change = max(abs(beliefs[g] - old_beliefs[g]) for g in all_genes)
            if max_change < tolerance:
                break
    
    return beliefs


def process_sample(sample_udp):
    """Process a single sample with factor graph BP"""
    pathway_activities = {}
    
    for pathway, interactions in pathway_interactions.items():
        # Build factor graph structure *within the worker*
        all_genes, gene_to_factors, factor_to_genes, factor_functions = build_factor_graph(interactions)
        
        initial_beliefs = {gene: sample_udp.get(gene, 0.5) for gene in all_genes}
        
        final_beliefs = loopy_belief_propagation_manual(
            all_genes, gene_to_factors, factor_to_genes, factor_functions,
            initial_beliefs
        )
        
        # Calculate pathway activity as mean belief
        if final_beliefs:
            pathway_activity = np.mean(list(final_beliefs.values()))
        else:
            pathway_activity = 0.5 # Default for empty pathway
        pathway_activities[pathway] = pathway_activity
    
    return pathway_activities


def calc_activity(udp_file=f'./data/{TEST}output_udp.csv',
                  output_file=f'./data/{TEST}output_activity.csv'):
    """
    Calculate pathway activities using factor graph belief propagation
    and the parallel_apply function.
    """   
    # Load UDP values
    udp_df = pd.read_csv(udp_file, index_col=0)
    udp_df.index = udp_df.index.str.lower()

    # 1. Create a 'func' for parallel_apply
    func_to_apply = process_sample
    
    # 2. Transpose udp_df so rows are samples, columns are genes
    df_to_process = udp_df.T
    
    print(f"Processing {len(df_to_process)} samples via parallel_apply...")
    
    # 3. Call parallel_apply
    #    It will iterate over rows of df_to_process (samples)
    #    It returns a new DataFrame where rows are samples
    #    and columns are pathway activities
    activity_df_T = parallel_apply(df_to_process, func_to_apply)
    
    # 4. Transpose back to match original activity_df format
    #    (rows=pathways, cols=samples)
    activity_df = activity_df_T.T
    
    # --- End of new logic ---

    # Reorder pathways to match original parsing
    activity_df = activity_df.reindex(pathway_names)
    
    # Save results
    activity_df.T.round(3).to_csv(output_file)
    print(f"\nSaved activity matrix to {output_file}")
    
    return activity_df


if __name__ == '__main__':
    calc_activity()