import pandas as pd
import numpy as np
from config import *
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
from itertools import product


# Using bnlearn.
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


def build_model_and_cpds(interactions, sample_udp, DEBUG=False):
    """
    Build a Bayesian network model of pathway interactions.
    Each interaction (I_k) is a node.
    Each gene connects I_parent -> I_child with P(I_child|I_parent) = UDP(gene),
    adjusted for inhibitory edges.
    """

    model = BayesianNetwork()
    edges = []
    child_parents = {}

    # Build edge list based on shared genes between interactions
    for i1, (src1, _, tgt1, _) in enumerate(interactions):
        for i2, (src2, _, tgt2, _) in enumerate(interactions):
            if i1 == i2:
                continue  # skip self
            shared = set(tgt1) & set(src2)
            for g in shared:
                edges.append((f"I_{i1}", f"I_{i2}", g))
                child_parents.setdefault(f"I_{i2}", []).append((f"I_{i1}", g))

    # Add all nodes upfront so they exist for cycle checks
    for i in range(len(interactions)):
        model.add_node(f"I_{i}")

    # Add edges safely, skipping cycles
    for parent, child, _ in edges:
        if parent == child:
            continue
        # Prevent cycles before adding
        if nx.has_path(model, child, parent):
            if DEBUG:
                print(f"⚠️ Skipping edge {parent} → {child} (would create loop)")
            continue
        model.add_edge(parent, child)

    if DEBUG:
        print("\n=== Pathway graph structure ===")
        for parent, child, gene in edges:
            print(f"{parent} → {child} via {gene}")
        print("===============================\n")

    # Build CPDs
    cpds = []
    for i, (src, typ, tgt, _) in enumerate(interactions):
        node = f"I_{i}"
        parents_info = child_parents.get(node, [])
        parents = [p for p, _ in parents_info if p != node]
        parents = list(dict.fromkeys(parents))  # deduplicate
    
        # keep only parents that are actual edges in model
        actual_edges = set(model.edges())
        parents = [p for p in parents if (p, node) in actual_edges]

        if not parents:
            # No parents: simple prior belief
            cpd = TabularCPD(variable=node, variable_card=2, values=[[0.5], [0.5]])
        else:
            n_parents = len(parents)
            n_cols = 2 ** n_parents

            probs = []
            for combo in product([0, 1], repeat=n_parents):
                p_up = 1.0
                for (parent, gene), val in zip(parents_info, combo):
                    # use UDP of gene (default 0.5 if missing)
                    udp_val = float(sample_udp.get(gene, 0.5))
                    # inhibitory edges invert probability
                    if "inhib" in typ.lower() or "dephosph" in typ.lower():
                        udp_val = 1.0 - udp_val
                    if val == 1:
                        p_up *= udp_val
                    else:
                        p_up *= (1 - udp_val)
                probs.append(p_up)

            values = np.vstack([1 - np.array(probs), np.array(probs)])  # shape (2, n_cols)

            # --- Robust normalization: avoid divide-by-zero and numerical drift ---
            col_sums = values.sum(axis=0, keepdims=True)
            # Replace any zeros with 1 to avoid division errors
            col_sums[col_sums == 0] = 1.0
            values = np.divide(values, col_sums, where=col_sums != 0)

            # Replace any NaN or Inf with uniform 0.5
            values = np.nan_to_num(values, nan=0.5, posinf=0.5, neginf=0.5)

            # Force exact normalization (sums = 1)
            values /= np.maximum(values.sum(axis=0, keepdims=True), 1e-12)

            # Clip to [0, 1] to remove residual rounding drift
            values = np.clip(values, 0.0, 1.0)

            # Re-normalize one last time to ensure exact column sums = 1
            values /= values.sum(axis=0, keepdims=True)


            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=values,
                evidence=parents,
                evidence_card=[2] * n_parents,
            )

        cpds.append(cpd)

    # Add all CPDs to model
    model.add_cpds(*cpds)

    # Validate the model
    try:
        model.check_model()
        if DEBUG:
            print("✅ Bayesian model structure and CPDs validated successfully.\n")
    except Exception as e:
        print(f"⚠️ Model validation failed: {e}")

    return model

def calc_pathway_activity(interactions, sample_udp):
    model = build_model_and_cpds(interactions, sample_udp)
    # check model is valid
    # use VariableElimination for exact inference
    infer = VariableElimination(model)
    # compute marginal P(node=up) for each interaction node
    probs = []
    for i in range(len(interactions)):
        node = f"I_{i}"
        q = infer.query([node], show_progress=False)
        # q is a DiscreteFactor: get probability for state 1 ('up') which is index 1
        val = q.get_value(**{node: 1})
        # val is array [P(down), P(up)]
        if val.size >= 2:
            p_up = float(val[1])
        else:
            p_up = 0.5
        probs.append(p_up)
    return float(np.mean(probs)) if probs else 0.5


def process_sample(sample_udp: pd.Series):
    pathway_interactions, pathway_names = parse_pathway_interactions('./data/pathway_relations.csv')
    activities = {}
    for path, inters in pathway_interactions.items():
        activities[path] = calc_pathway_activity(inters, sample_udp)
        if DEBUG:
            break
    return activities


def calc_activity(udp_file='./data/output_udp.csv', output_file='./data/output_activity.csv'):
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()
    if udp_df.max().max() < 25:
        udp_df = 2 ** udp_df
    if DEBUG:
        udp_df = udp_df.iloc[:, :1]
    df_to_process = udp_df.T
    results = parallel_apply(df_to_process, process_sample)
    results = pd.DataFrame(results.tolist(), index=df_to_process.index)
    results.to_csv(output_file)
    return results


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')