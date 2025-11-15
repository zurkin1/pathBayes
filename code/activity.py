'''
Electrical flow treats the graph as an electrical circuit:
- Each edge has a conductance (capacity), here = UDP.
- Sources are fixed at high potential; sinks are fixed at low potential.
- All other node potentials are determined by Kirchhoff’s law (net current = 0 at interior nodes).
- Solving these constraints leads to a linear system of the form 𝐿𝑣=𝑏, where 𝐿 is the graph Laplacian.
- Once node potentials 𝑣 are known, flows = conductance × voltage-difference on each edge.
- This is not solving an ODE over time - it’s solving a static linear system that gives the unique steady-state currents in one shot.
'''
import pandas as pd
import networkx as nx
import warnings
from metrics import *
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


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


# Initialize graph structures. Built once at module load, reused for all samples.
pathway_interactions, pathway_names = parse_pathway_interactions('./data/pathway_relations.csv')


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


def initialize_pathway_graphs(PATHWAY_GRAPHS):
    """Build and cache all pathway graph structures once"""
    print("Building pathway graph structures...")
    for pathway, interactions in pathway_interactions.items():
        PATHWAY_GRAPHS[pathway] = build_pathway_graph_structure(interactions)
    print(f"Built {len(PATHWAY_GRAPHS)} pathway graphs")


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def resistance_pathway_activity(sample_udp: pd.Series, G):
    """Using NetworkX. Requires undirected graph."""
    G_undirected = G.to_undirected()
    # Set edge weights as conductances
    for u, v, data in G_undirected.edges(data=True):
        genes = data['genes']
        # Parallel conductances add up
        total_conductance = sum(sample_udp.get(gene, 0.0) for gene in genes)
        G_undirected[u][v]['conductance'] = total_conductance
    
    # Get pre computed corridors
    corridors = G.graph.get('corridors', [])
    
    # Compute resistance only for identified corridors
    total_flow = 0.0
    for source, sink in corridors:
        # Get connected component containing this source-sink pair
        component_nodes = nx.node_connected_component(G_undirected, source)
        if sink in component_nodes:
            subgraph = G_undirected.subgraph(component_nodes).copy()              
            resistance = nx.resistance_distance(
                subgraph, source, sink, 
                weight='conductance',
                invert_weight=False
            )
            
            if resistance > 0:
                flow = 1.0 / resistance
                if is_inhibitory(G.nodes[sink].get("interaction_type", "")):
                    flow = -flow
                total_flow += flow
                    
    return float(total_flow)


def process_sample(sample_udp: pd.Series, PATHWAY_GRAPHS):
    """
    Compute pathway activities for one sample.
    """
    activities = {}
    for pathway, G in PATHWAY_GRAPHS.items():
        activity = resistance_pathway_activity(sample_udp, G)
        activities[pathway] = activity
    return activities


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


def calc_activity(udp_file='./data/output_udp.csv', output_file='./data/output_activity.csv'):
    """Main entry: load UDP, run pathway analysis, and save activity matrix."""
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()

    PATHWAY_GRAPHS = {}
    initialize_pathway_graphs(PATHWAY_GRAPHS)

    DEBUG=False
    if DEBUG:
        for col in udp_df.columns:
            print(f"Processing sample {col}...")
            result = process_sample(udp_df[col])
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