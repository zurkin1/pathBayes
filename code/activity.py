import pandas as pd
import numpy as np
from config import *
import networkx as nx
import warnings
from metrics import *


warnings.simplefilter("error", RuntimeWarning) #Stop on warnings.


# Load and parse interactions into simple pathway_interactions dictionary data structure. e.g.
# pathway_interactions['Adherens junction'][0] = (['baiap2', 'wasf2', 'wasf3', 'wasf1'], 'activation', ['actb', 'actg1'], 'Adherens junction')
def parse_pathway_interactions(relations_file):
    """Parse interactions and assign unique IDs to each"""
    pathway_relations = pd.read_csv(relations_file)
    pathway_relations['source'] = pathway_relations['source'].fillna('').astype(str).str.lower().str.split('*')
    pathway_relations['target'] = pathway_relations['target'].fillna('').astype(str).str.lower().str.split('*')   
    
    interactions_by_pathway = {}
    for idx, row in pathway_relations.iterrows():
        pathway = row['pathway']
        if pathway not in interactions_by_pathway:
            interactions_by_pathway[pathway] = []
        
        # Store interaction with its global ID for fast lookup
        interactions_by_pathway[pathway].append({
            'id': idx,  # Unique interaction ID
            'source': row['source'],
            'type': row['interactiontype'],
            'target': row['target'],
            'pathway': pathway
        })
    
    return interactions_by_pathway, list(interactions_by_pathway.keys())


pathway_interactions, pathway_names = parse_pathway_interactions('./data/pathway_relations.csv')


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def build_pathway_graph_structure(interactions):
    """
    Build the static graph structure for a pathway.
    This is called once per pathway and cached.
    Stores only topology and gene names no sample-specific data.
    Breaks cycles using edge betweenness centrality.
    
    Returns: NetworkX graph with:
    - Nodes: interaction IDs
    - Node attrs: source_genes, target_genes, interaction_type
    - Edge attrs: gene (the shared gene creating this edge)
    """
    G = nx.MultiDiGraph()
    
    # Add all nodes first
    for interaction in interactions:
        i_id = interaction['id']
        G.add_node(
            i_id,
            source_genes=interaction['source'],
            target_genes=interaction['target'],
            interaction_type=interaction['type']
        )
    
    # Create edges based on gene sharing (target of i1 → source of i2)
    for int1 in interactions:
        for int2 in interactions:
            if int1['id'] == int2['id']:
                continue
            
            shared_genes = set(int1['target']) & set(int2['source'])
            for gene in shared_genes:
                G.add_edge(
                    int1['id'], 
                    int2['id'], 
                    gene=gene # Store which gene creates this connection
                )

    # Break cycles using edge betweenness centrality
    max_iterations = 1000 # Safety limit to avoid infinite loops
    iteration = 0
    
    while not nx.is_directed_acyclic_graph(G) and iteration < max_iterations:
        iteration += 1
        
        try:
            # Find a cycle
            cycle = nx.find_cycle(G, orientation='original')
            
            # Extract edges involved in this cycle
            cycle_edges = [(u, v, k) for u, v, k, _ in cycle]
            
            # Calculate edge betweenness for the entire graph
            # edge_betweenness_centrality doesn't support MultiDiGraph directly
            # So we need to handle multi-edges separately
            edge_betweenness = {}
            
            # For MultiDiGraph, calculate betweenness on simplified graph
            # then apply to all parallel edges
            G_simple = nx.DiGraph(G) # Collapse multi-edges
            betweenness_simple = nx.edge_betweenness_centrality(G_simple)
            
            # Map betweenness to multi-edges
            for u, v, k in cycle_edges:
                edge_betweenness[(u, v, k)] = betweenness_simple.get((u, v), 0)
            
            # Remove edge with minimum betweenness. For multi edges it will remove (36, 82, 0), then on the next loop (36, 82, 1) and so on.
            min_edge = min(edge_betweenness.items(), key=lambda x: x[1])[0]
            G.remove_edge(min_edge[0], min_edge[1], min_edge[2])
            
        except nx.NetworkXNoCycle:
            # No cycle found, we're done
            break
    
    if iteration >= max_iterations:
        print(f"Warning: Could not break all cycles after {max_iterations} iterations")

    return G


def initialize_pathway_graphs(PATHWAY_GRAPHS):
    """Build and cache all pathway graph structures once"""
    print("Building pathway graph structures...")
    for pathway, interactions in pathway_interactions.items():
        PATHWAY_GRAPHS[pathway] = build_pathway_graph_structure(interactions)
    print(f"Built {len(PATHWAY_GRAPHS)} pathway graphs")


def calculate_node_belief(node_id, G, sample_udp):
    """
    Calculate belief for a node on-the-fly based on current sample.
    """
    src_genes = G.nodes[node_id]['source_genes']
    tgt_genes = G.nodes[node_id]['target_genes']
    itype = G.nodes[node_id]['interaction_type']
    
    # Compute source and target activities from sample
    src_activity = sum(sample_udp.get(g, 0.0) for g in src_genes)
    tgt_activity = sum(sample_udp.get(g, 0.0) for g in tgt_genes)
    tgt_activity = max(1e-10, tgt_activity)
    
    # Scale ratio
    belief = gaussian_scaling(src_activity, tgt_activity)
    if is_inhibitory(itype):
        belief = -belief
    
    return belief


def calculate_edge_udp(gene, sample_udp, is_inhibitory_edge):
    """Calculate UDP for an edge based on the gene and sample"""
    udp_orig = np.clip(float(sample_udp.get(gene, 0.0)), 1e-6, 1 - 1e-6)
    udp = to_prob_power(udp_orig)
    if is_inhibitory_edge:
        udp = 1 - udp
    return udp


def update_belief_optimized(G, node, sample_udp, node_beliefs):
    """
    Update belief for a node using cached graph structure.
    All sample-specific values (UDP, beliefs) are computed on-the-fly.
    """
    min_v = sample_udp.min()
    max_v = sample_udp.max()
    
    def normalize(x):
        return (x - min_v) / (max_v - min_v + 1e-9)
    
    parents = list(G.predecessors(node))
    
    if not parents:
        return # No parents, keep initial belief
    
    scaled = []
    for parent in parents:
        scale = 1.0 / len(parents)
        belief_p = node_beliefs.get(parent, scale)
        
        # Get all edges from parent to node (multi-edge support)
        edges_data = G[parent][node]
        scaled_p_belief = []
        
        for edge_key, edge_attr in edges_data.items():
            gene = edge_attr['gene']
            
            # Check if parent's interaction type is inhibitory
            parent_itype = G.nodes[parent]['interaction_type']
            is_inhib = is_inhibitory(parent_itype)
            
            # Calculate UDP on-the-fly
            udp = calculate_edge_udp(gene, sample_udp, is_inhib)
            udp_normalized = normalize(udp)
            
            scaled_p_belief.append(belief_p * udp_normalized)
        
        scaled.append(np.mean(scaled_p_belief) * scale)
    
    # Noisy-OR aggregation
    if scaled:
        B_orig = node_beliefs.get(node, 0.5)
        B = 1 - np.prod([1 - s for s in scaled])
        alpha = 0.5
        node_beliefs[node] = alpha * B + (1 - alpha) * B_orig


def process_sample(sample_udp: pd.Series, PATHWAY_GRAPHS):
    """
    Compute pathway activity using CACHED graph structures.
    Only sample-specific computations happen here.
    """
    activities = {}
    
    for pathway, G in PATHWAY_GRAPHS.items():
        # Initialize beliefs for this sample
        node_beliefs = {}
        for node in G.nodes:
            node_beliefs[node] = calculate_node_belief(node, G, sample_udp)
        
        # Update beliefs via belief propagation
        for node in nx.topological_sort(G):
            update_belief_optimized(G, node, sample_udp, node_beliefs)
        
        # Average all node beliefs for pathway activity
        activities[pathway] = float(np.mean(list(node_beliefs.values())))
    
    return activities


def calc_activity(udp_file='./data/output_udp.csv', output_file='./data/output_activity.csv'):
    """Main entry: load UDP, run pathway analysis, and save activity matrix."""
    
    # Initialize graph structures.
    # Global cache for pathway graphs.
    # Built once at module load, reused for all samples
    PATHWAY_GRAPHS = {}
    initialize_pathway_graphs(PATHWAY_GRAPHS)
    
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()
    
    if DEBUG:
        for col in udp_df.columns:
            print(f"Processing sample {col}...")
            result = process_sample(udp_df[col], PATHWAY_GRAPHS)
        exit(0)
    
    df_to_process = udp_df.T
    print(f"Processing {len(df_to_process)} samples...")
    results = parallel_apply(df_to_process, process_sample, PATHWAY_GRAPHS).T
    results = results.round(4)
    results.to_csv(output_file)
    print(f"Saved results to {output_file}")
    return results


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')