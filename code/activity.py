import pandas as pd
import numpy as np
from config import *
import networkx as nx
import random


# Load and parse interactions into simple pathway_interactions dictionary data structure. e.g.
# pathway_interactions['Adherens junction'][0] = (['baiap2', 'wasf2', 'wasf3', 'wasf1'], 'activation', ['actb', 'actg1'], 'Adherens junction')
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
    """
    Build a directed graph of interaction nodes (I_1, I_2, ...)
    where edges (genes) connect parent interactions to child interactions.
    Genes serve as evidence carriers, not nodes.
    """
    G = nx.DiGraph()
    factor_map = {} # target gene -> interaction ID(s)
    factor_id = 0

    # Step 1: Create interaction nodes
    for src_genes, itype, tgt_genes, _ in interactions:
        if not tgt_genes:
            continue
        is_inhib = is_inhibitory(itype)
        node_id = f"I_{factor_id}"
        factor_id += 1
        G.add_node(node_id, sources=src_genes, targets=tgt_genes, type=itype, is_inhib=is_inhib)
        for tgt in tgt_genes:
            factor_map.setdefault(tgt, []).append(node_id)

    # Step 2: Connect interactions via shared genes
    for i_node, attrs in G.nodes(data=True):
        for g in attrs["targets"]:
            # Find all interactions that use this gene as source
            for j_node, j_attrs in G.nodes(data=True):
                if g in j_attrs["sources"]:
                    G.add_edge(i_node, j_node, gene=g, type=j_attrs["type"])

    return G


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


def belief_propagation_interaction_graph(G, gene_priors, max_iter=30, tolerance=1e-3, update_fraction=0.3):
    """
    Propagate beliefs over an interaction graph (NetworkX DiGraph).
    Each node = interaction (activation/inhibition),
    Each edge = gene carrying evidence.
    gene_priors: dict {gene_name: expression_value in [0,1]}
    Stochastic belief propagation: at each iteration, randomly update a subset of edges/nodes.
    update_fraction: fraction of nodes to update each iteration (0–1).
    """
    beliefs = {n: 0.5 for n in G.nodes}

    # Assign edge-level "effect" (+1 activation, -1 inhibition)
    for u, v, data in G.edges(data=True):
        itype = data.get("type", "")
        data["effect"] = -1 if is_inhibitory(itype) else +1

    for iteration in range(max_iter):
        old_beliefs = beliefs.copy()

        # Randomly pick subset of nodes to update
        nodes_to_update = random.sample(
            list(G.nodes), max(1, int(update_fraction * len(G.nodes)))
        )

        for node in nodes_to_update:
            incoming_values = []

            # Collect influences from parents
            for pred in G.predecessors(node):
                edge_data = G[pred][node]
                gene = edge_data["gene"]
                gene_val = gene_priors.get(gene, 0.5)
                edge_sign = edge_data["effect"]

                # influence = parent_belief * gene_val * sign
                influence = edge_sign * beliefs[pred] * gene_val
                incoming_values.append(influence)

            # If no parents, use direct evidence from source genes
            if not incoming_values:
                src_genes = G.nodes[node]["sources"]
                incoming_values = [gene_priors.get(g, 0.5) for g in src_genes]

            # Combine incoming influences (sum then squash to [0,1])
            combined = np.mean(incoming_values) if incoming_values else 0.5
            combined = np.clip(combined, 0.0, 1.0)

            # Compute updated belief (edge-level modulation already applied)
            output = CPT_BASELINE + (CPT_ACTIVATION - CPT_BASELINE) * combined
            beliefs[node] = np.clip(output, 0.0, 1.0)

        # Convergence check
        max_change = max(abs(beliefs[n] - old_beliefs[n]) for n in nodes_to_update)
        if max_change < tolerance:
            break

    return beliefs


def process_sample(sample_udp: pd.Series) -> dict[str, float]:
    """Process a single sample and return pathway activities."""
    pathway_activities = {}

    for pathway, interactions in pathway_interactions.items():
        G = build_factor_graph_structure(interactions)
        if len(G.nodes) == 0:
            pathway_activities[pathway] = 0.5
            continue

        # Use expression values as priors
        gene_priors = {g: sample_udp.get(g, 0.5) for g in set(sample_udp.index)}

        # Run Bayesian propagation on the graph
        beliefs = belief_propagation_interaction_graph(G, gene_priors)

        # Pathway activity = mean belief across interactions
        pathway_activities[pathway] = float(np.mean(list(beliefs.values())))

    return pathway_activities


def calc_activity(udp_file=f'./data/output_udp.csv',
                  output_file=f'./data/output_activity.csv'):
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