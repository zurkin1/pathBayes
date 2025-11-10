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


def initialize_node_belief(interaction, sample_udp):
    """
    Initialize a node's belief (activity) based on its genes.
    """
    belief = 0.0
    src_activity, tgt_activity = 0.0, 0.0
    (src_genes, itype, tgt_genes, _) = interaction

    # Compute input (source) activity
    for g in src_genes:
        if g in sample_udp:
            src_activity += sample_udp[g]

    # Compute output (target) activity
    for g in tgt_genes:
        if g in sample_udp:
            tgt_activity += sample_udp[g]
    tgt_activity = max(1e-10, tgt_activity)

    # Scale ratio
    belief = gaussian_scaling(src_activity, tgt_activity)
    if is_inhibitory(itype):
        belief =  -belief # inverse for inhibitory

    return belief


def build_bayesnet(interactions, sample_udp):
    """
    Build a BayesNet for a single pathway using sorobn.
    Each interaction (I_k) is a node; edges connect interactions via genes.
    For edge I1 -> I2 via gene g, we set:
        P(I2 | I1=True)  = UDP(g)
        P(I2 | I1=False) = CPT_BASELINE
    """
    G = nx.MultiDiGraph() #We can have multiple edges between two interactions.

    # Create edges between interactions via shared genes
    for i1, (src1, t1, tgt1, _) in enumerate(interactions):
        for i2, (src2, t2, tgt2, _) in enumerate(interactions):
            if not i1 in G:
                belief = initialize_node_belief((src1, t1, tgt1, _), sample_udp)
                G.add_node(i1, belief=belief)
            if not i2 in G:
                belief = initialize_node_belief((src2, t2, tgt2, _), sample_udp)
                G.add_node(i2, belief=belief)
            if i1 == i2:
                continue
            shared = set(tgt1) & set(src2)
            for g in shared:
                udp_orig = np.clip(float(sample_udp.get(g, 0.0)), 1e-6, 1 - 1e-6)
                udp = to_prob_power(udp_orig)
                if is_inhibitory(t2):
                    udp = 1 - udp
                G.add_edge(i1, i2, gene=g, udp=udp) # Keep udp as attribute.

    return G


def update_belief(G, node, sample_udp):
    min_v, max_v = np.min(sample_udp), np.max(sample_udp)
    
    
    #Normalize
    def nor(x):
         return (x - min_v) / (max_v - min_v + 1e-9)
    
    
    parents = list(G.predecessors(node))

    # Collect scaled parent contributions
    scaled = []
    for p in parents:
        scale = 1/len(parents)
        belief_p = G.nodes[p].get('belief', scale) # default neutral belief
        #Loop over all edges between nodes node and p e.g.:{0: {'gene': 'TP53', 'udp': 0.8}, 1: {'gene': 'EGFR', 'udp': 0.5}}.
        scaled_p_belief = []
        for key, attr in G[p][node].items():
            udp = nor(attr.get("udp"))
            scaled_p_belief.append(belief_p*udp)
        scaled.append(np.mean(scaled_p_belief) * scale)
        
    # If the node has parents use noisy-OR aggregation: P = 1 - ∏(1 - scaled_i).
    if scaled:
        B_orig = G.nodes[node].get('belief')
        B = 1 - np.prod([1 - s for s in scaled])  
        alpha = 0.5
        G.nodes[node]['belief'] = alpha*B + (1-alpha)*B_orig


def process_sample(sample_udp: pd.Series):
    """Compute pathway activity by updating beliefs of root nodes via noisy-OR,
    then propagating and averaging all node beliefs."""
    activities = {}

    for path, interactions in pathway_interactions.items():
        G = build_bayesnet(interactions, sample_udp)
        #Update beliefs P(node=True) of all nodes.
        for node in G.nodes: #Already sorted in topological order.
            update_belief(G, node, sample_udp)

        activities[path] = float(np.mean([G.nodes[p].get('belief', 0.5) for p in G.nodes]))

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
    results = results.round(4)
    results.to_csv(output_file)
    print(f"Saved results to {output_file}")
    return results


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')