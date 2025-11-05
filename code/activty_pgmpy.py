import pandas as pd
import numpy as np
from config import *
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import itertools


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


def process_sample(sample_udp):
    """Process a single sample with factor graph BP using pgmpy"""
    pathway_activities = {}
    
    for pathway, interactions in pathway_interactions.items():
        # Get all genes in the pathway
        all_genes = set()
        for sources, _, targets, _ in interactions:
            all_genes.update(sources)
            all_genes.update(targets)
        
        if not all_genes:
            pathway_activities[pathway] = 0.5
            continue
        
        # Initial beliefs (priors)
        initial_beliefs = {gene: sample_udp.get(gene, 0.5) for gene in all_genes}
        
        # Create FactorGraph
        fg = FactorGraph()
        fg.add_nodes_from(all_genes)
        
        # Add unary prior factors for each gene
        for gene in all_genes:
            prior_p1 = initial_beliefs[gene]
            prior_values = [1.0 - prior_p1, prior_p1]  # P(0), P(1)
            prior_factor = DiscreteFactor([gene], [2], prior_values)
            fg.add_factor(prior_factor)
        
        # Add interaction factors
        for idx, (sources, inttype, targets, _) in enumerate(interactions):
            is_inhib = is_inhibitory(inttype)
            
            for target in targets:
                if not sources:
                    continue  # Skip if no sources
                
                variables = sources + [target]
                card = [2] * len(variables)
                
                # Generate all possible assignments
                num_vars = len(variables)
                values = []
                for combo in itertools.product([0, 1], repeat=num_vars):
                    source_states = combo[:-1]
                    target_state = combo[-1]
                    
                    combined = 1.0 - np.prod([1.0 - s for s in source_states])
                    
                    if is_inhib:
                        output = CPT_BASELINE * (1.0 - CPT_INHIBITION * combined)
                    else:
                        output = CPT_BASELINE + CPT_ACTIVATION * combined
                    
                    output = np.clip(output, 0.0, 1.0)
                    
                    if target_state == 1:
                        phi = output
                    else:
                        phi = 1.0 - output
                    
                    values.append(phi)
                
                factor = DiscreteFactor(variables, card, values)
                fg.add_factor(factor)
        
        # Perform loopy belief propagation
        bp = BeliefPropagation(fg, max_iter=30, convergence_threshold=1e-3)
        bp.calibrate()
        
        # Get final beliefs (P(active=1) for each gene)
        final_beliefs = {}
        for gene in all_genes:
            marginal = bp.query(variables=[gene])
            final_beliefs[gene] = marginal.values[1]  # P(1)
        
        # Calculate pathway activity as mean belief
        if final_beliefs:
            pathway_activity = np.mean(list(final_beliefs.values()))
        else:
            pathway_activity = 0.5
        
        pathway_activities[pathway] = pathway_activity
    
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