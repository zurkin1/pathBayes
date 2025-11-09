import pandas as pd
import numpy as np
from config import *


# Load and parse interactions into simple dictionary data structure.
def parse_pathway_interactions(relations_file):
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


pathway_interactions, pathway_names = parse_pathway_interactions(f'./data/pathway_relations.csv')


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def build_factor_graph_structure(interactions):
    """Build factor graph structure with explicit interaction nodes."""
    all_genes = set()
    for src, _, tgt, _ in interactions:
        all_genes.update(src)
        all_genes.update(tgt)
    
    # Identify input vs target genes
    target_genes = set()
    for src, _, tgt, _ in interactions:
        target_genes.update(tgt)
    input_genes = all_genes - target_genes
    
    # Build interaction factors
    factors = []
    factor_id = 0
    
    for src_list, itype, tgt_list, _ in interactions:
        is_inhib = is_inhibitory(itype)
        
        for target in tgt_list:
            if not src_list:
                continue
                
            src_list = list(set(src_list))
            src_list_filtered = [s for s in src_list if s != target]
            
            if len(src_list_filtered) > 0:
                factors.append({
                    'id': f'factor_{factor_id}',
                    'sources': src_list_filtered,
                    'target': target,
                    'is_inhibitory': is_inhib
                })
                factor_id += 1
    
    return all_genes, input_genes, target_genes, factors


def compute_factor_output(source_beliefs, is_inhibitory):
    """Compute output of a factor given source beliefs."""
    # Noisy-OR combination of sources
    combined = 1.0 - np.prod([1.0 - b for b in source_beliefs])
    
    # Apply CPT
    if is_inhibitory:
        output = CPT_BASELINE * (1.0 - CPT_INHIBITION * combined)
    else:
        output = CPT_BASELINE + CPT_ACTIVATION * combined
    
    return np.clip(output, 0.0, 1.0)


def belief_propagation_factor_graph(all_genes, input_genes, factors, initial_beliefs,
                                    max_iter=30, tolerance=1e-3):
    """
    Run belief propagation on factor graph with explicit interaction nodes.
    
    Factor graph structure:
    - Gene nodes: all genes
    - Factor nodes: one per interaction
    - Messages: gene->factor, factor->gene
    """
    # Initialize beliefs
    beliefs = {g: initial_beliefs.get(g, 0.5) for g in all_genes}
    
    # Initialize messages
    # gene_to_factor[gene][factor_id] = message
    # factor_to_gene[factor_id] = message to target gene
    gene_to_factor = {g: {} for g in all_genes}
    factor_to_gene = {f['id']: 0.5 for f in factors}
    
    # Build reverse index: which factors does each gene send to?
    gene_sends_to_factors = {g: [] for g in all_genes}
    for factor in factors:
        for src in factor['sources']:
            gene_sends_to_factors[src].append(factor['id'])
    
    # Build index: which factors send to each gene?
    gene_receives_from_factors = {g: [] for g in all_genes}
    for factor in factors:
        gene_receives_from_factors[factor['target']].append(factor['id'])
    
    # Main BP loop
    for iteration in range(max_iter):
        old_beliefs = beliefs.copy()
        
        # PHASE 1: Gene to Factor messages
        for gene in all_genes:
            for factor_id in gene_sends_to_factors[gene]:
                if gene in input_genes:
                    # Input genes always send their prior
                    gene_to_factor[gene][factor_id] = initial_beliefs.get(gene, 0.5)
                else:
                    # Non-input genes: combine incoming messages from OTHER factors
                    incoming = [factor_to_gene[f] for f in gene_receives_from_factors[gene]]
                    if incoming:
                        # Product of incoming factor messages with prior
                        msg = np.prod(incoming) * initial_beliefs.get(gene, 0.5)
                        gene_to_factor[gene][factor_id] = np.clip(msg, 0.0, 1.0)
                    else:
                        gene_to_factor[gene][factor_id] = initial_beliefs.get(gene, 0.5)
        
        # PHASE 2: Factor to Gene messages
        for factor in factors:
            # Collect messages from source genes
            source_beliefs = [gene_to_factor[src].get(factor['id'], beliefs[src]) 
                             for src in factor['sources']]
            
            # Compute factor output
            output = compute_factor_output(source_beliefs, factor['is_inhibitory'])
            factor_to_gene[factor['id']] = output
        
        # PHASE 3: Update gene beliefs
        for gene in all_genes:
            if gene in input_genes:
                # Input genes keep their prior
                beliefs[gene] = initial_beliefs.get(gene, 0.5)
            else:
                # Target genes: combine all incoming factor messages
                incoming = [factor_to_gene[f] for f in gene_receives_from_factors[gene]]
                
                if incoming:
                    # Product of factor messages with prior
                    prior = initial_beliefs.get(gene, 0.5)
                    combined = np.prod(incoming) * prior
                    beliefs[gene] = np.clip(combined, 0.0, 1.0)
                else:
                    # No incoming factors (shouldn't happen for target genes)
                    beliefs[gene] = initial_beliefs.get(gene, 0.5)
        
        # Check convergence
        max_change = max(abs(beliefs[g] - old_beliefs[g]) for g in all_genes)
        if max_change < tolerance:
            break
    
    return beliefs


def process_sample(sample_udp: pd.Series) -> dict[str, float]:
    """Process a single sample and return pathway activities."""
    pathway_activities = {}
    
    for pathway, interactions in pathway_interactions.items():
        # Build factor graph structure
        all_genes, input_genes, target_genes, factors = build_factor_graph_structure(interactions)
        
        if not all_genes:
            pathway_activities[pathway] = 0.5
            continue
        
        # Get priors for this sample
        initial_beliefs = {g: sample_udp.get(g, 0.5) for g in all_genes}
        
        # Run belief propagation
        final_beliefs = belief_propagation_factor_graph(
            all_genes, input_genes, factors, initial_beliefs
        )
        
        # Compute pathway activity
        pathway_activities[pathway] = float(np.mean(list(final_beliefs.values())))
    
    return pathway_activities


def calc_activity(udp_file=f'./data/{TEST}output_udp.csv',
                  output_file=f'./data/{TEST}output_activity.csv'):
    """
    Calculate pathway activities using factor graph belief propagation
    and the parallel_apply function.
    """   
    # Load UDP values
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
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
    
    # Reorder pathways to match original parsing
    #activity_df = activity_df.reindex(pathway_names)
    
    # Save results
    activity_df.T.round(3).to_csv(output_file)
    print(f"\nSaved activity matrix to {output_file}")
    
    return activity_df


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')