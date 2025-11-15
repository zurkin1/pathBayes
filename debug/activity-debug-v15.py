'''
aximum flow treats the graph as a network:
- Each edge has a capacity (UDP expression level).
- Sources are starting points; sinks are ending points.
- Maximum flow computes the maximum amount of "signal" that can flow from source to sink.
- This directly models pathway activity as signal propagation capacity.
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
import numpy as np
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse import csr_matrix


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
    
    return interactions_by_pathway


def build_pathway_graph_structure(interactions):
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
            if shared_genes:
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

    #Remove acycles
    while True:
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(*cycle[0][:2])
        except nx.exception.NetworkXNoCycle:
            break

    # Find top 20 longest paths using iterative removal
    corridors = []
    G_temp = G.copy()
    for _ in range(20):
        try:
            path = nx.dag_longest_path(G_temp)
        except (nx.NetworkXError, nx.NetworkXNotImplemented):
            break
        if len(path) < 2:
            break
        corridors.append((path[0], path[-1]))
        G_temp.remove_edges_from(list(zip(path, path[1:])))
        if G_temp.number_of_edges() == 0:
            break
    
    G.graph['corridors'] = corridors
    return G


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def compute_max_flow_scipy(G, source, sink, sample_udp, scale_factor=1000, debug=False):
    """
    Compute maximum flow using scipy's sparse graph implementation.
    Returns the maximum flow value from source to sink.
    
    Args:
        scale_factor: Scale float capacities to integers (default 1000 for good precision)
        debug: Print debugging information
    """
    # Create node index mapping
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # Build capacity matrix
    capacity = np.zeros((n, n), dtype=np.int32)
    for u, v, data in G.edges(data=True):
        genes = data['genes']
        # Sum UDP values for all genes creating this edge
        total_capacity = sum(sample_udp.get(gene, 0.0) for gene in genes)
        # Scale to integer while preserving precision
        capacity[node_to_idx[u], node_to_idx[v]] = np.int32(total_capacity * scale_factor)
    
    # Convert to sparse matrix with explicit integer dtype
    capacity_sparse = csr_matrix(capacity, dtype=np.int32)
    
    # Compute maximum flow
    source_idx = node_to_idx[source]
    sink_idx = node_to_idx[sink]
    
    if debug:
        print(f"\n=== DEBUG: source={source}, sink={sink} ===")
        print(f"Source idx: {source_idx}, Sink idx: {sink_idx}")
        print(f"Total nodes: {n}, Total edges: {G.number_of_edges()}")
        
        # Check if path exists
        try:
            path = nx.shortest_path(G, source, sink)
            print(f"Shortest path exists: {len(path)} nodes")
            print(f"Path: {path[:5]}..." if len(path) > 5 else f"Path: {path}")
            
            # Check capacities along path
            print("\nCapacities along shortest path:")
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                cap = capacity[node_to_idx[u], node_to_idx[v]]
                genes = G[u][v]['genes']
                udp_sum = sum(sample_udp.get(gene, 0.0) for gene in genes)
                print(f"  {u} -> {v}: capacity={cap}, UDP_sum={udp_sum:.4f}, genes={genes}")
        except nx.NetworkXNoPath:
            print("NO PATH EXISTS between source and sink!")
        
        # Check non-zero capacities
        nonzero = np.count_nonzero(capacity)
        print(f"\nNon-zero capacities in matrix: {nonzero}")
        print(f"Min non-zero capacity: {capacity[capacity > 0].min() if nonzero > 0 else 'N/A'}")
        print(f"Max capacity: {capacity.max()}")
    
    flow = maximum_flow(capacity_sparse, source_idx, sink_idx)
    
    if debug:
        print(f"Flow value (scaled): {flow.flow_value}")
        print(f"Flow value (original): {flow.flow_value / scale_factor}")
    
    # Scale back to original units
    return flow.flow_value / scale_factor


def process_sample(sample_udp: pd.Series):
    debug=True
    """Compute pathway activities for one sample using maximum flow."""
    global PATHWAY_GRAPHS
    activities = {}
    
    for pathway, G in PATHWAY_GRAPHS.items():
        corridors = G.graph.get('corridors', [])
        if not corridors:
            activities[pathway] = 0.0
            continue
        
        if debug:
            print(f"\n{'='*60}")
            print(f"PATHWAY: {pathway}")
            print(f"Corridors: {len(corridors)}")
        
        # Compute maximum flow for each corridor
        total_flow = 0.0
        valid_corridors = 0
        zero_flow_count = 0
        
        for idx, (source, sink) in enumerate(corridors):
            try:
                # Enable debug for first corridor or when flow is zero
                flow_value = compute_max_flow_scipy(G, source, sink, sample_udp, debug=debug and idx < 2)
                if flow_value > 0:
                    total_flow += flow_value
                    valid_corridors += 1
                else:
                    zero_flow_count += 1
                    if debug and idx < 5:
                        print(f"*** ZERO FLOW for corridor {idx}: {source} -> {sink}")
            except Exception as e:
                if debug:
                    print(f"ERROR in corridor {idx} ({source} -> {sink}): {e}")
                continue
        
        if debug:
            print(f"\nSummary for {pathway}:")
            print(f"  Valid corridors with flow > 0: {valid_corridors}")
            print(f"  Zero flow corridors: {zero_flow_count}")
            print(f"  Total flow: {total_flow}")
        
        # Average flow across valid corridors
        if valid_corridors > 0:
            activities[pathway] = float(total_flow / valid_corridors)
        else:
            activities[pathway] = 0.0
    
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
    pathway_interactions = parse_pathway_interactions('./data/pathway_relations.csv')

    print("Building pathway graph structures...")
    for pathway, interactions in pathway_interactions.items():
        PATHWAY_GRAPHS[pathway] = build_pathway_graph_structure(interactions)
    print(f"Built {len(PATHWAY_GRAPHS)} pathway graphs")
    
    udp_df = pd.read_csv('./data/TCGACRC_expression-merged.zip', sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()

    DEBUG=True
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