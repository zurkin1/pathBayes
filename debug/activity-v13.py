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


def build_pathway_graph_structure(pathway, interactions):
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

    # ---------------------------------------
    # Reduce pathway to acyclic corridor once
    # ---------------------------------------
    corridor_edges = get_acyclic_corridor_edges(G, pathway, k=5)
    G.graph['corridor_edges'] = len(corridor_edges) > 5
    SG = G.edge_subgraph(corridor_edges).copy() #Build MultiDiGraph SG that contains only the nodes and the specified (u,v,key) edges.
    num_corridors = nx.number_weakly_connected_components(SG)
    #print(f"[Corridor] Pathway '{pathway}' → {num_corridors} corridors ({len(corridor_edges)} edges)")
    if len(corridor_edges) > 5:
        G = SG
    else:
        pass
        #print(f"[Warning] Pathway '{pathway}' has no corridor edges. Using full graph.")

    # ---------------------------------------------------------
    # PRUNE: keep only nodes on at least one source → sink path
    # ---------------------------------------------------------
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    sinks   = [node for node in G.nodes if G.out_degree(node) == 0]
    # 1) forward reachability from sources
    reachable_from_sources = set()
    for s in sources:
        reachable_from_sources |= nx.descendants(G, s)
        reachable_from_sources.add(s)

    # 2) backward reachability to sinks
    can_reach_sinks = set()
    for t in sinks:
        can_reach_sinks |= nx.ancestors(G, t)
        can_reach_sinks.add(t)

    # 3) intersection = true conductive backbone
    valid_nodes = reachable_from_sources & can_reach_sinks

    # If nothing remains, no valid corridor → shallow fallback
    if not valid_nodes:
        return G

    # 4) prune graph
    G = G.subgraph(valid_nodes).copy()

    return G


def shallow_fallback_activity(G, sample_udp):
    """
    Old-style node belief: gaussian scaling of sum(incoming UDP) and sum(outgoing UDP)
    Returns average over nodes.
    """
    vals = []
    for node in G.nodes:
        src_genes = G.nodes[node].get("source_genes", [])
        tgt_genes = G.nodes[node].get("target_genes", [])

        src_sum = sum(sample_udp.get(g, 0.0) for g in src_genes)
        tgt_sum = sum(sample_udp.get(g, 0.0) for g in tgt_genes)
        tgt_sum = max(tgt_sum, 1e-10)

        b = gaussian_scaling(src_sum, tgt_sum)
        if is_inhibitory(G.nodes[node].get("interaction_type", "")):
            b = -b
        vals.append(b)

    return float(np.mean(vals)) if vals else 0.0


def initialize_pathway_graphs(PATHWAY_GRAPHS):
    """Build and cache all pathway graph structures once"""
    print("Building pathway graph structures...")
    for pathway, interactions in pathway_interactions.items():
        PATHWAY_GRAPHS[pathway] = build_pathway_graph_structure(pathway, interactions)
    print(f"Built {len(PATHWAY_GRAPHS)} pathway graphs")


def get_acyclic_corridor_edges(G, pathway, k=5, multiplier=3, max_cycle_break=1000):
    """
    Select a corridor of important edges (by edge-betweenness) and ensure it is acyclic.
    Returns a set of (u, v, key) for edges to keep.
    - k: desired number of 'core' edges (approx).
    - multiplier: initially select up to k*multiplier candidate edges to give room for cycle-breaking.
    """
    # 1) compute edge betweenness on a collapsed simple DiGraph
    H = nx.DiGraph()
    for u, v, key in G.edges(keys=True):
        H.add_edge(u, v)
    bet = nx.edge_betweenness_centrality(H)

    # 2) choose top candidate pairs (u,v)
    pairs_sorted = sorted(bet.items(), key=lambda x: x[1], reverse=True)
    top_n = min(len(pairs_sorted), max(1, k * multiplier))
    candidate_pairs = [p for p, _ in pairs_sorted[:top_n]]
    if not candidate_pairs:
        return set()

    # 3) map back to multi-edge keys and collect candidate edges
    candidate_edges = {(u, v, k) for (u, v) in candidate_pairs for k in G[u][v]}

    # 4) Build initial corridor subgraph SG containing only candidate edges
    SG = G.edge_subgraph(candidate_edges).copy()

    # 5) If SG has cycles, break them by removing the lowest-betweenness edge in the cycle
    iteration = 0
    while not nx.is_directed_acyclic_graph(SG) and iteration < max_cycle_break:
        iteration += 1
        try:
            cycle = nx.find_cycle(SG, orientation='original')
        except nx.NetworkXNoCycle:
            break

        # cycle is list of (u, v, key, dir); pick the edge with smallest betweenness (on collapsed pair)
        cycle_edges = [(u, v, k) for u, v, k, _ in cycle]
        # compute betweenness for each cycle edge using the collapsed value (u,v)
        eb_values = {e: bet.get((e[0], e[1]), 0.0) for e in cycle_edges}

        # remove the cycle edge with minimum betweenness from SG (only that specific parallel edge)
        edge_to_remove = min(eb_values.items(), key=lambda x: x[1])[0]
        u, v, key = edge_to_remove
        if SG.has_edge(u, v, key):
            SG.remove_edge(u, v, key)
            # Also remove from candidate_edges set to keep consistency
            candidate_edges.discard((u, v, key))
        else:
            # fallback: remove any one parallel edge between u and v present in SG
            keys = list(SG[u][v].keys())
            if keys:
                SG.remove_edge(u, v, keys[0])
                candidate_edges = {e for e in candidate_edges if not (e[0]==u and e[1]==v and e[2]==keys[0])}

    # 6) After cycle-breaking SG should be acyclic (or we hit max iterations).
    # Return the final kept edges as (u,v,key) tuples.
    kept = set()
    for u, v, key, data in SG.edges(keys=True, data=True):
        kept.add((u, v, key))
    return kept


def resistance_pathway_activity(G, pathway, sample_udp):
    """
    Electrical-flow (Kirchhoff) pathway activity.
    Edges = conductances (UDP). Sources fixed at 1, sinks fixed at 0.
    Solve L v = b for interior node potentials v, then compute total current
    entering sinks. That current = pathway activity.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}

    # --- 1. Build conductance matrix C (edge weights = UDP) ---
    C = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        gene = data['gene']
        udp = sample_udp.get(gene, 0.0)
        if udp < 0:
            udp = 0.0
        C[node_index[u], node_index[v]] += udp

    # If no edges carry conductance → no flow → activity = 0
    if C.sum() == 0:
        print(f'Pathway {pathway} no edges carry conductance.')
        return 0.0

    # --- 2. Identify sources (roots) and sinks (leaves) ---
    sources = [node for node in nodes if G.in_degree(node) == 0]
    sinks   = [node for node in nodes if G.out_degree(node) == 0]

    if len(sources) == 0 or len(sinks) == 0:
        # No meaningful flow
        print(f'Pathway {pathway} no meaningful flow.')
        return 0.0

    source_idx = [node_index[s] for s in sources]
    sink_idx   = [node_index[t] for t in sinks]

    # --- 3. Build the graph Laplacian L ---
    # L = D - C  (D = out-degree conductance)
    D = np.diag(C.sum(axis=1))
    L = D - C

    # --- 4. Prepare boundary conditions: sources at 1.0, sinks at 0.0 ---
    # Dirichlet boundary: sources=1, sinks=0 → no internal current injection
    fixed = set(source_idx + sink_idx)

    # Split into free vs fixed nodes
    free_idx   = [i for i in range(n) if i not in fixed]
    fixed_idx  = list(fixed)

    # Build right-hand side for interior nodes:
    # L_ff * v_f + L_fF * v_F = 0  →  L_ff v_f = -L_fF v_F
    L_ff = L[np.ix_(free_idx, free_idx)]
    L_fF = L[np.ix_(free_idx, fixed_idx)]

    v_fixed = np.zeros(len(fixed_idx))
    for j, idx in enumerate(fixed_idx):
        # sources at 1, sinks at 0
        if idx in source_idx:
            v_fixed[j] = 1.0
        else:
            v_fixed[j] = 0.0

    rhs = -L_fF @ v_fixed

    # --- 5. Solve for interior node potentials. Use pseudoinverse. ---
    try:
        v_free = np.linalg.solve(L_ff, rhs)
        #v_free = np.linalg.lstsq(L_ff, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        print(f'Pathway {pathway} singular matrix.')
        return 0.0 # singular, no flow possible

    # Reconstruct full potential vector v
    v = np.zeros(n)
    for j, idx in enumerate(free_idx):
        v[idx] = v_free[j]
    for j, idx in enumerate(fixed_idx):
        v[idx] = v_fixed[j]

    # --- 6. Compute net current entering sinks ---
    total_flow = 0.0
    for t in sinks:
        ti = node_index[t]
        # current = sum over incoming edges: conductance * (v_u - v_t)
        inflow = 0.0
        for u in G.predecessors(t):
            ui = node_index[u]
            conductance = C[ui, ti]
            inflow += conductance * (v[ui] - v[ti])   # (v[t] = 0)

        # Flip sign if this interaction is inhibitory
        if is_inhibitory(G.nodes[t].get("interaction_type", "")):
            inflow = -inflow

        total_flow += inflow

    return float(total_flow)


def process_sample(sample_udp: pd.Series, PATHWAY_GRAPHS):
    """
    Compute pathway activities for one sample.
    """
    activities = {}
    for pathway, G in PATHWAY_GRAPHS.items():
        if G.graph['corridor_edges']:
            activity = resistance_pathway_activity(G, pathway, sample_udp)
        else:
            activity = shallow_fallback_activity(G, sample_udp)
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
    
    if True:
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