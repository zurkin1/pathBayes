'''
Electrical flow treats the graph as an electrical circuit:
- Each edge has a conductance (capacity), here = UDP.
- Sources are fixed at high potential; sinks are fixed at low potential.
- All other node potentials are determined by Kirchhoff’s law (net current = 0 at interior nodes).
- Solving these constraints leads to a linear system of the form 𝐿𝑣=𝑏, where 𝐿 is the graph Laplacian.
- Once node potentials 𝑣 are known, flows = conductance × voltage-difference on each edge.
- This is not solving an ODE over time - it’s solving a static linear system that gives the unique steady-state currents in one shot.
'''
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import pandas as pd
import networkx as nx
import warnings
from metrics import *
import multiprocessing as mp
from tqdm import tqdm
import random
import numpy as np


warnings.simplefilter("error", RuntimeWarning) #Stop on warnings.


def parse_pathway_interactions(relations_file):
    """
    Parse interactions and assign unique IDs to each
    Load and parse interactions into simple pathway_interactions dictionary data structure. e.g.
    pathway_interactions['Adherens junction'][0] = (['baiap2', 'wasf2', 'wasf3', 'wasf1'], 'activation', ['actb', 'actg1'], 'Adherens junction')
    """
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
            'id': idx,
            'source': row['source'],
            'type': row['interactiontype'],
            'target': row['target'],
            'pathway': pathway
        })
    
    return interactions_by_pathway, list(interactions_by_pathway.keys())


def build_pathway_graph_structure(pathway, interactions):
    """
    Build the static graph structure for a pathway.
    Stores only topology and gene names no sample-specific data.
    
    Returns: NetworkX graph with:
    - Nodes: interaction IDs
    - Node attrs: source_genes, target_genes, interaction_type
    - Edge attrs: genes (the shared genes creating this edge)
    - Graph attrs: corridors (list of (source, sink) tuples for top 20 longest paths)
    """
    G = nx.DiGraph()
    
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
            if shared_genes and shared_genes != {''}:
                # Store all genes that create this connection
                if G.has_edge(int1['id'], int2['id']):
                    # Add to existing gene list
                    G[int1['id']][int2['id']]['genes'].update(shared_genes)
                else:
                    # Create new edge
                    G.add_edge(int1['id'], int2['id'], genes=shared_genes)

    # Prune: keep only nodes on at least one source → sink path
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    sinks   = [node for node in G.nodes if G.out_degree(node) == 0]
    if sources and sinks:
        reachable_from_sources = set()
        for s in sources:
            reachable_from_sources |= nx.descendants(G, s) | {s}
        
        can_reach_sinks = set()
        for t in sinks:
            can_reach_sinks |= nx.ancestors(G, t) | {t}
        
        valid_nodes = reachable_from_sources & can_reach_sinks
        if valid_nodes:
            G = G.subgraph(valid_nodes).copy()

    def make_acyclic(G):
        G_copy = G.copy()
        try:
            while True:
                cycle = nx.find_cycle(G_copy, orientation='original')
                # Remove one edge from the cycle to break it
                edge_to_remove = cycle[0][:2]
                G_copy.remove_edge(*edge_to_remove)
        except nx.exception.NetworkXNoCycle:
            # No cycles left
            pass
        return G_copy

    # Find top 20 longest paths using iterative removal
    def longest_path_top_k(G, k=20):
        paths = []
        G_copy = G.copy()
        for _ in range(k):
            try:
                longest_path = nx.dag_longest_path(G_copy)
            except (nx.NetworkXError, nx.NetworkXNotImplemented):
                break
            if len(longest_path) < 2:
                break
            paths.append(longest_path)
            edges_to_remove = list(zip(longest_path, longest_path[1:]))
            G_copy.remove_edges_from(edges_to_remove)
            if G_copy.number_of_edges() == 0:
                break
        return paths
    
    G_acyclic = make_acyclic(G)
    # Get top 20 longest paths and extract (source, sink) pairs
    longest_paths = longest_path_top_k(G_acyclic, k=20)
    corridors = [(path[0], path[-1]) for path in longest_paths if len(path) >= 2]
    G.graph['corridors'] = corridors
    
    return G


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def batch_resistance(G, pathway, pairs):
    """
    Adaptive resistance distance computation:
    - Small components (<20 nodes): compute all pairs at once
    - Large components with many pairs (>30%): compute all at once  
    - Large components with few pairs: sample 10 random pairs and extrapolate
    """    
    n_nodes = G.number_of_nodes()
    n_pairs = len(pairs)
    
    # Strategy 1: Small component - compute all pairwise. Strategy 2: Large component with many pairs.
    if (n_nodes < 20) or (n_pairs > 0.3 * n_nodes * (n_nodes - 1) / 2):
        all_resistances = nx.resistance_distance(G, weight='conductance', invert_weight=False)
        
        results = []
        for source, sink in pairs:
            resistance = all_resistances[source][sink]
            results.append(resistance)
    else:    
        # Strategy 3: Large component with few pairs - sample up to 10
        random.seed(42)
        sample_size = min(10, n_pairs)
        
        # Randomly select which pairs to compute
        if sample_size < n_pairs:
            sampled_indices = sorted(random.sample(range(n_pairs), sample_size))
            sampled_pairs = [pairs[i] for i in sampled_indices]
        else:
            sampled_indices = list(range(n_pairs))
            sampled_pairs = pairs
        
        # Compute resistances for sampled pairs
        results = []
        for source, sink in sampled_pairs:
            resistance = nx.resistance_distance(G, source, sink, weight='conductance', invert_weight=False)
            results.append(resistance)
    
    return np.mean(results)


def process_sample(sample_udp: pd.Series):
    """Compute pathway activities for one sample using NetworkX. Requires undirected graph."""
    global PATHWAY_GRAPHS
    activities = {}
    for pathway, G in PATHWAY_GRAPHS.items():
        # Get pre computed corridors
        corridors = G.graph.get('corridors', [])
        if not corridors:
            activities[pathway] = 0.0
            continue

        G_undirected = nx.Graph()
        # Set edge weights as conductances
        for u, v, data in G.edges(data=True):
            genes = data['genes']
            # Parallel conductances add up
            total_conductance = sum(sample_udp.get(gene, 0.0) for gene in genes)
            #if total_conductance > 0: # Skip zero-conductance edges
            G_undirected.add_edge(u, v, conductance=total_conductance)

        # Group corridors by connected component to avoid redundant work
        component_map = {}
        for node in G_undirected.nodes():
            if node not in component_map:
                component = frozenset(nx.node_connected_component(G_undirected, node))
                for n in component:
                    component_map[n] = component

        # Process corridors grouped by component
        corridors_by_component = {}
        for source, sink in corridors:
            if source in component_map and sink in component_map:
                component = component_map[source]
                if component_map[sink] == component: # Same component
                    if component not in corridors_by_component:
                        corridors_by_component[component] = []
                    corridors_by_component[component].append((source, sink))

        # Adaptive batch compute resistances per component
        total_flow = 0.0
        len_comp = len(corridors_by_component.items())
        for component, corridor_pairs in corridors_by_component.items():
            subgraph = G_undirected.subgraph(component)
            
            # Use adaptive strategy
            resistance = max(1.0, batch_resistance(subgraph, pathway, corridor_pairs))
            flow = 1.0 / resistance # We are calculating current I=V/R.
            #if is_inhibitory(G.nodes[sink].get("interaction_type", "")):
            #    flow = -flow
            total_flow += flow
        
        activities[pathway] = float(total_flow/len_comp)

    return activities


def parallel_apply(df):
    """Applies a function to DataFrame rows in parallel, preserving order."""
    n_cores = max(1, mp.cpu_count() - 2) # leave 2 cores free for OS
    with mp.Pool(n_cores) as pool:
        results = list(
            tqdm(
                pool.imap(process_sample, (row for _, row in df.iterrows())),
                total=len(df),
            )
        )
    return pd.DataFrame(results, index=df.index)


PATHWAY_GRAPHS = {}


if __name__ == '__main__':
    # Use fork on Linux it's faster and handles complex objects
    #if mp.get_start_method(allow_none=True) != 'fork':
    #    mp.set_start_method('fork', force=True)
    # Initialize graph structures. Built once at module load, reused for all samples.
    pathway_interactions, pathway_names = parse_pathway_interactions('./data/pathway_relations.csv')

    print("Building pathway graph structures...")
    for pathway, interactions in pathway_interactions.items():
        PATHWAY_GRAPHS[pathway] = build_pathway_graph_structure(pathway, interactions)
    print(f"Built {len(PATHWAY_GRAPHS)} pathway graphs")
    
    
    udp_df = pd.read_csv('./data/TCGACRC_expression-merged.zip', sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()

    DEBUG=False
    if DEBUG:
        for col in udp_df.columns:
            print(f"Processing sample {col}...")
            result = process_sample(udp_df[col])
        exit(0)
   
    df_to_process = udp_df.T
    print(f"Processing {len(df_to_process)} samples...")
    results = parallel_apply(df_to_process).T
    results = results.round(4)
    results.to_csv('./data/output_activity.csv')
    print(f"Saved results to ./data/output_activity.csv")