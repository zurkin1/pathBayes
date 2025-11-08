import pandas as pd
import numpy as np
from config import *
from bayes_net import BayesNet
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
    nodes, edges, inhibitory = [], [], []

    # Create nodes for all interactions.
    for idx in range(len(interactions)):
        src_genes, itype, tgt_genes, _ = interactions[idx]
        node = f"I_{idx}"
        nodes.append(node)
        if is_inhibitory(itype):
            inhibitory.append(node)

    # Create edges between interactions via shared genes
    for i1, (src1, t1, tgt1, _) in enumerate(interactions):
        for i2, (src2, t2, tgt2, _) in enumerate(interactions):
            shared = set(tgt1) & set(src2)
            for g in shared:
                edges.append((f"I_{i1}", f"I_{i2}", g))

    # Master root node that connects to all real root nodes.
    all_targets = {v for _, v, _ in edges}
    root_nodes = [n for n in nodes if n not in all_targets]
    nodes.append("I_1000")

    for root in root_nodes:
        for src_genes, _, tgt_genes, _ in interactions:
            for g in src_genes:
                edges.append(("I_1000", root, g))

    # Add edges: keep only valid (u, v) pairs of distinct strings
    G = nx.DiGraph()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 3:
            u, v, g = e
            if isinstance(u, str) and isinstance(v, str) and u != v:
                G.add_edge(u, v, gene=g) # Keep gene as attribute.

    # --- Detect and remove cycles ---
    try:
        cycle = list(nx.find_cycle(G, orientation='original'))
        while len(cycle) > 0:
            u, v, _ = cycle[0]
            #if DEBUG:
            #    print(f"⚠️ Removing edge {u}->{v} to break cycle.")
            G.remove_edge(u, v)
            cycle = list(nx.find_cycle(G, orientation='original'))
    except nx.NetworkXNoCycle:
        pass

    # Initialize network
    bn = BayesNet(*( (u, v) for u, v in G.edges ), seed=42)

    # Assign CPTs for nodes in proper Series format.
    #Assume conditional independence of causes given the child (as in Noisy-OR).
    T, F = True, False
    for v in G.nodes:
        parents = list(G.predecessors(v))
        if not parents:
            #Master node is always True. We configure dangling nodes to true as well to use their UDP directly.
            bn.P[v] = pd.Series({True: 1.0, False: 0.0})
            continue

        cpt = {}
        n = len(parents)

        # baseline: all parents False
        all_false = tuple([F] * n + [T])
        cpt[all_false] = CPT_BASELINE
        cpt[tuple([F] * n + [F])] = 1 - CPT_BASELINE

        # one-parent active cases
        for i, p in enumerate(parents):
            g = G[p][v].get("gene", "")
            udp_orig = np.clip(float(sample_udp.get(g, 0.5)), 1e-6, 1 - 1e-6)
            udp = to_prob_power(udp_orig)
            if v in inhibitory:
                udp = 1 - udp
            state_true = [F] * n
            state_true[i] = T
            cpt[tuple(state_true + [T])] = udp
            cpt[tuple(state_true + [F])] = 1 - udp

        bn.P[v] = pd.Series(cpt)

    bn.prepare()
    return bn, G


def process_sample(sample_udp: pd.Series):
    """Compute pathway activity by updating beliefs of root nodes via noisy-OR,
    then propagating and averaging all node beliefs."""
    activities = {}

    for path, interactions in pathway_interactions.items():
        bn, G = build_bayesnet(interactions, sample_udp)

        #Update_beliefs_via_sampling.
        n_samples=100
        samples = bn.sample(n_samples)
        bn.fit(samples)

        # --- Query beliefs of all nodes ---
        probs = []
        for idx, node in enumerate(bn.nodes):
            q = bn.query(node, event={}) #, algorithm='likelihood'
            probs.append(q.mean())

        activities[path] = float(np.mean(probs)) if probs else 0.0
        if DEBUG:
            break # only one for debugging

    return activities


def calc_activity(udp_file='./data/output_udp.csv', output_file='./data/output_activity.csv'):
    """Main entry: load UDP, run pathway analysis, and save activity matrix."""
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()
    if DEBUG:
        for col in udp_df.columns:
            print(f"Processing sample {col}...")
            process_sample(udp_df[col])
        exit(0)

    df_to_process = udp_df.T
    print(f"Processing {len(df_to_process)} samples...")
    results = parallel_apply(df_to_process, process_sample).T
    results.to_csv(output_file)
    print(f"Saved results to {output_file}")
    return results


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')