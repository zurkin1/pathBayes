import pandas as pd
import numpy as np
from config import *
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, FactorGraph, BeliefPropagation
import itertools
from typing import List, Tuple, Dict


def parse_pathway_interactions(relations_file):
    """Loads and parses pathway relations into a dict."""
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


# Load and parse interactions into simple data structures
pathway_interactions, pathway_names = parse_pathway_interactions(f'./data/{TEST}pathway_relations.csv')


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation',
                           'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


# ----------------------------------------------------------------------
# 2. Build a pomegranate FactorGraph for ONE pathway
# ----------------------------------------------------------------------
def build_pathway_factor_graph(
    interactions: List[Tuple[List[str], str, List[str], str]],
    evidence: Dict[str, float]
) -> Tuple[FactorGraph, Dict[str, int]]:
    """
    Returns (factor_graph, node_index_map)
    node_index_map: gene -> integer index used by pomegranate
    """
    # ---- collect all genes ------------------------------------------------
    all_genes = set()
    for srcs, _, tgts, _ in interactions:
        all_genes.update(srcs)
        all_genes.update(tgts)
    genes = sorted(all_genes)
    n = len(genes)
    idx = {g: i for i, g in enumerate(genes)}

    # ---- create the graph ------------------------------------------------
    fg = FactorGraph()

    # ---- prior distributions (evidence) ----------------------------------
    for gene in genes:
        p1 = evidence.get(gene, 0.5)               # P(active=1)
        dist = DiscreteDistribution({0: 1.0 - p1, 1: p1})
        fg.add_node(dist, name=gene)

    # ---- interaction factors (noisy-OR + CPT) ----------------------------
    for src_list, itype, tgt_list, _ in interactions:
        is_inhib = is_inhibitory(itype)

        for target in tgt_list:
            if not src_list:
                continue

            src_idxs = [idx[s] for s in src_list]
            tgt_idx = idx[target]

            # Build CPT: every combination of source states → P(target)
            table = []
            for assignment in itertools.product([0, 1], repeat=len(src_list)):
                combined = 1.0 - np.prod([1.0 - s for s in assignment])   # noisy-OR

                if is_inhib:
                    out = CPT_BASELINE * (1.0 - CPT_INHIBITION * combined)
                else:
                    out = CPT_BASELINE + CPT_ACTIVATION * combined
                out = np.clip(out, 0.0, 1.0)

                # P(target=0) and P(target=1)
                table.append([*assignment, 0, 1.0 - out])
                table.append([*assignment, 1, out])

            # pomegranate expects variables in the *same* order as the table
            vars_in_order = src_idxs + [tgt_idx]
            cpt = ConditionalProbabilityTable(table, [fg.nodes[i] for i in src_idxs])
            fg.add_factor(cpt, connects=vars_in_order)

    return fg, idx


# ----------------------------------------------------------------------
# 3. Process ONE sample
# ----------------------------------------------------------------------
def process_sample(sample_udp: pd.Series) -> Dict[str, float]:
    pathway_activities: Dict[str, float] = {}

    for pathway, interactions in pathway_interactions.items():
        # ---- evidence (gene → P(active)) ---------------------------------
        all_genes = {g for srcs, _, tgts, _ in interactions
                     for g in srcs + tgts}
        evidence = {g: sample_udp.get(g, 0.5) for g in all_genes}

        if not all_genes:
            pathway_activities[pathway] = 0.5
            continue

        # ---- build graph --------------------------------------------------
        fg, node_idx = build_pathway_factor_graph(interactions, evidence)

        # ---- loopy BP -----------------------------------------------------
        bp = BeliefPropagation(fg)
        bp.loopy(max_iterations=30, tolerance=1e-3)

        # ---- marginals ----------------------------------------------------
        marginals = {
            gene: bp.belief(node_idx[gene])[1]      # P(gene=1)
            for gene in all_genes
        }

        pathway_activities[pathway] = float(np.mean(list(marginals.values())))

    return pathway_activities


def calc_activity(udp_file=f'./data/{TEST}output_udp.csv',
                  output_file=f'./data/{TEST}output_activity.csv'):
    """
    Calculate pathway activities using factor graph belief propagation
    and the parallel_apply function.
    """   
    # Load UDP values
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()

    # 1. Create a 'func' for parallel_apply
    func_to_apply = process_sample
    
    # 2. Transpose udp_df so rows are samples, columns are genes
    df_to_process = udp_df.T
    
    print(f"Processing {len(df_to_process)} samples via parallel_apply...")
    
    # 3. Call parallel_apply
    #    It will iterate over rows of df_to_process (samples)
    #    It returns a new DataFrame where rows are samples
    #    and columns are pathway activities
    activity_df_T = parallel_apply(df_to_process, func_to_apply)
    
    # 4. Transpose back to match original activity_df format
    #    (rows=pathways, cols=samples)
    activity_df = activity_df_T.T
    
    # Reorder pathways to match original parsing
    #activity_df = activity_df.reindex(pathway_names)
    
    # Save results
    activity_df.T.round(3).to_csv(output_file)
    print(f"\nSaved activity matrix to {output_file}")
    
    return activity_df


if __name__ == '__main__':
    calc_activity('./data/TCGACRC_expression-merged.zip')