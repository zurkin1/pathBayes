import pandas as pd
import numpy as np
import networkx as nx
import warnings
from metrics import *
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


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
    return G


def initialize_pathway_graphs(PATHWAY_GRAPHS):
    """Build and cache all pathway graph structures once"""
    print("Building pathway graph structures...")
    for pathway, interactions in pathway_interactions.items():
        PATHWAY_GRAPHS[pathway] = build_pathway_graph_structure(interactions)
    print(f"Built {len(PATHWAY_GRAPHS)} pathway graphs")


def diffuse_pathway_activity(G, sample_udp, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Diffusion-based pathway activity calculation.
    Edges carry UDP-derived weights (normalized), representing diffusion capacity.
    Nodes = interactions; edges = shared genes.
    Steps:
    1) Assign edge weights = normalized UDP of connecting gene.
    2) Build stochastic transition matrix W.
    3) Run random-walk-with-restart until steady state.
    4) Return mean steady-state activation of nodes (optionally leaves only).
    """
    nodes = list(G.nodes)
    n = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}

    # Build weighted adjacency matrix W based on gene UDP values
    W = np.zeros((n, n), dtype=float)
    min_v, max_v = sample_udp.min(), sample_udp.max()
    norm = lambda x: (x - min_v) / (max_v - min_v + 1e-9)

    for u, v, data in G.edges(data=True):
        gene = data['gene']
        udp = norm(sample_udp.get(gene, 0.0))
        # optional: handle inhibitory edges
        if is_inhibitory(G.nodes[u]['interaction_type']):
            udp = -udp
        W[node_index[u], node_index[v]] += udp

    # Normalize to make stochastic matrix
    W = W / (W.max() + 1e-9)

    # Restart vector: use the sample’s UDP values aggregated over genes belonging to each node.
    s = np.zeros(n)
    for node in nodes:
        src_genes = G.nodes[node]['source_genes']
        tgt_genes = G.nodes[node]['target_genes']
        node_udp = np.mean([sample_udp.get(g, 0.0) for g in src_genes + tgt_genes])
        s[node_index[node]] = node_udp
    # Normalize only to prevent explosion, not to erase contrast
    s /= (np.max(s) + 1e-9)

    # Random walk with restart
    x = s.copy()
    for _ in range(max_iter):
        x_new = alpha * W @ x + (1 - alpha) * s
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new

    # Average node activations as pathway activity
    return float(np.mean(x))


def process_sample(sample_udp: pd.Series, PATHWAY_GRAPHS):
    """
    Compute diffusion-based pathway activities for one sample.
    """
    activities = {}
    for pathway, G in PATHWAY_GRAPHS.items():
        activity = diffuse_pathway_activity(G, sample_udp)
        activities[pathway] = activity
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


def parallel_apply(df, func, PATHWAY_GRAPHS):
    """Applies a function to DataFrame rows in parallel, preserving order."""
    n_cores = max(1, mp.cpu_count() - 2) # leave 2 cores free for OS
    func_with_pathway = partial(func, PATHWAY_GRAPHS=PATHWAY_GRAPHS)
    with mp.Pool(n_cores) as pool:
        results = list(
            tqdm(
                pool.imap(func_with_pathway, [row for _, row in df.iterrows()]),
                total=len(df),
                desc="Processing samples",
            )
        )

    return pd.DataFrame(results, index=df.index)


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')