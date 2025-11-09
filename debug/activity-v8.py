import pandas as pd
import numpy as np
from config import *
import sorobn
import networkx as nx


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


def build_bayesnet(interactions, sample_udp):
    """
    Build a BayesNet for a single pathway using sorobn.
    Each interaction (I_k) is a node; edges connect interactions via genes.
    For edge I1 -> I2 via gene g, we set:
        P(I2 | I1=True)  = UDP(g)
        P(I2 | I1=False) = CPT_BASELINE
    """
    nodes, edges = [], []

    # Create nodes for all interactions
    for i, (src_genes, itype, tgt_genes, _) in enumerate(interactions):
        if tgt_genes:
            nodes.append(f"I_{i}")

    # Create edges between interactions via shared genes
    for i1, (src1, t1, tgt1, _) in enumerate(interactions):
        for i2, (src2, t2, tgt2, _) in enumerate(interactions):
            shared = set(tgt1) & set(src2)
            for g in shared:
                edges.append((f"I_{i1}", f"I_{i2}", g))

    # Clean edges: keep only valid (u, v) pairs of distinct strings
    G = nx.DiGraph()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 3:
            u, v, g = e
            if isinstance(u, str) and isinstance(v, str) and u != v:
                # Assign CPTs for each child node in proper Series format.
                #Assume conditional independence of causes given the child (as in Noisy-OR).
                udp_orig = float(sample_udp.get(g, 0.5))
                udp = to_prob_power(udp_orig)
                G.add_edge(u, v, gene=g, udp=udp) # Keep gene as attribute.

    # --- Detect and remove cycles ---
    try:
        cycle = list(nx.find_cycle(G, orientation='original'))
        while len(cycle) > 0:
            u, v, _ = cycle[0]
            print(f"⚠️ Removing edge {u}->{v} to break cycle.")
            G.remove_edge(u, v)
            cycle = list(nx.find_cycle(G, orientation='original'))
    except nx.NetworkXNoCycle:
        pass

    # Initialize network
    bn = sorobn.BayesNet(*( (u, v) for u, v in G.edges ))

    T, F = True, False
    #bn.P[v] = pd.Series({(T, T): udp, # P(v=True | u=True) (T, F): 1 - udp, # P(v=False | u=True) (F, T): CPT_BASELINE, # P(v=True | u=False) (F, F): 1 - CPT_BASELINE })

    # Set priors for root nodes (no parents)
    #for n in bn.nodes:
    #    bn.P[n] = pd.Series({
    #        True: to_prob_power(sample_udp.get(n, 0)),  # or any normalization you use
    #        False: 1 - to_prob_power(sample_udp.get(n, 0))
    #    })

    bn.prepare()
    return bn


def noisy_or_prob(bn, u, event, base=1e-3):
    """Compute P(node=True | event) using per-edge Noisy-OR on-the-fly."""
    parents = bn.parents(u)
    if not parents:
        # prior for root nodes
        incoming = [(a, b, d) for a, b, d in bn.graph.edges(data=True) if b == u]
        if not incoming:
            return 0.5 # fallback prior

        strengths = []
        for _, _, d in incoming:
            udp = float(d.get("udp", 0.5))
            udp = to_prob_power(udp)
            g = d.get("gene", "")
            if is_inhibitory(g):
                udp = 1 - udp
            strengths.append(np.clip(udp, 1e-6, 1 - 1e-6))

        p_true = 1 - np.prod([(1 - s) for s in strengths])
        return np.clip(p_true, 1e-6, 1 - 1e-6)

    #active_parents = [p for p in parents if event.get(p, False)] # we assume all parents in the event.
    strengths = [bn.graph[p][u].get("udp", 0.5) for p in parents]
    if strengths:
        #Noisy or assumption.
        p_true = 1 - np.prod([(1 - to_prob_power(s)) for s in strengths])
    else:
        p_true = base
    return np.clip(p_true, 1e-6, 1 - 1e-6)


def process_sample(sample_udp: pd.Series):
    """Compute all pathway activities for a given patient sample."""
    activities = {}
    for path, interactions in pathway_interactions.items():
        #activities[path] = calc_pathway_activity(inters, sample_udp)
        """Compute mean posterior probability across all interactions in a pathway."""
        bn = build_bayesnet(interactions, sample_udp)

        # Fit with pseudo-data. Finalizes internal structures (like normalization and message tables).
        df = pd.DataFrame([{k: sample_udp.get(k, 0) for k in bn.nodes}])
        bn.fit(df)

        # Compute activity as mean posterior probability
        probs = []
        for node in bn.nodes:
            try:
                #q = bn.query(node, event={})
                q = noisy_or_prob(bn, node, event={})
                probs.append(q)
            except Exception:
                continue

        activities[path] = float(np.mean(probs)) if probs else 0.0
        if DEBUG: break # only one for debugging
    return activities


def calc_activity(udp_file='./data/output_udp.csv', output_file='./data/output_activity.csv'):
    """Main entry: load UDP, run pathway analysis, and save activity matrix."""
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()
    if DEBUG:
        udp_df = udp_df.iloc[:, :1]
        print(f"DEBUG: Limiting UDP to one sample ({udp_df.columns[0]})")

    df_to_process = udp_df.T
    print(f"Processing {len(df_to_process)} samples...")
    results = parallel_apply(df_to_process, process_sample).T
    results.to_csv(output_file)
    print(f"Saved results to {output_file}")
    return results


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')