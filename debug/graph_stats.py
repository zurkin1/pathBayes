import pandas as pd
import networkx as nx
from tabulate import tabulate
import numpy as np
from config import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time


INPUT_FILE = "./data/pathway_relations.csv"


def parse_genes(s):
    if pd.isna(s) or s == '':
        return []
    return str(s).lower().split('*')


def build_graph(subdf):
    """Build directed MultiDiGraph exactly like activity.py."""
    G = nx.MultiDiGraph()
    for i, row in subdf.iterrows():
        G.add_node(i, source=row["source"], target=row["target"])
    for i1, r1 in subdf.iterrows():
        for i2, r2 in subdf.iterrows():
            if i1 == i2:
                continue
            shared = set(r1["target"]) & set(r2["source"])
            for g in shared:
                G.add_edge(i1, i2, gene=g)
    return G


def average_path_length(G):
    """Compute average length of all root→leaf paths in DAG."""
    if not nx.is_directed_acyclic_graph(G):
        return np.nan
    lengths = []
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    for r in roots:
        for l in leaves:
            try:
                for path in nx.all_simple_paths(G, r, l):
                    lengths.append(len(path) - 1)
            except nx.NetworkXNoPath:
                continue
    return np.mean(lengths) if lengths else 0.0


def analyze_pathway(pathway, subdf):
    G = build_graph(subdf)
    n_nodes = G.number_of_nodes()

    # 1) percentage of disconnected nodes (isolated)
    n_disconnected = len([n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0])
    perc_disconnected = 100 * n_disconnected / n_nodes if n_nodes else 0

    # 2) largest connected component size (undirected view)
    if n_nodes > 0:
        largest_cc = max(len(c) for c in nx.connected_components(G.to_undirected()))
    else:
        largest_cc = 0

    # 3) average path length (roots→leaves)
    avg_path_len = average_path_length(G)

    # 4) number of nodes
    num_nodes = n_nodes

    # 5) number of unique genes in pathway
    all_genes = set(g for row in subdf["source"] for g in row) | set(g for row in subdf["target"] for g in row)
    num_genes = len(all_genes)

    # 6) average number of source genes per interaction
    avg_sources = np.mean([len(row) for row in subdf["source"]]) if num_nodes else 0

    # 7) average number of target genes per interaction
    avg_targets = np.mean([len(row) for row in subdf["target"]]) if num_nodes else 0

    # 8) largest 'source' component (max len of source list)
    largest_source = max(len(row) for row in subdf["source"]) if num_nodes else 0

    # 9) largest 'target' component (max len of target list)
    largest_target = max(len(row) for row in subdf["target"]) if num_nodes else 0

    return [
        pathway,
        f"{perc_disconnected:.1f}%",
        largest_cc,
        round(avg_path_len, 2),
        num_nodes,
        num_genes,
        round(avg_sources, 2),
        round(avg_targets, 2),
        largest_source,
        largest_target
    ]


def main():
    df = pd.read_csv(INPUT_FILE)
    df["source"] = df["source"].apply(parse_genes)
    df["target"] = df["target"].apply(parse_genes)

    results = []
    for pathway, subdf in df.groupby("pathway"):
        if pathway in ['Cellular senescence', 'Circadian entrainment', 'Longevity regulating pathway - multiple species', 'NBreast cancer', 'NChemokine signaling pathway', 'NcAMP signaling pathway']:
            continue
        print(time.ctime(), f'Analyzing pathway: {pathway}')
        stats = analyze_pathway(pathway, subdf)
        results.append(stats)

    headers = [
        "Pathway",
        "% Disconnected",
        "Largest CC Size",
        "Avg Path Len",
        "# Nodes",
        "# Genes",
        "Avg # Src Genes",
        "Avg # Tgt Genes",
        "Largest Src Comp",
        "Largest Tgt Comp"
    ]

    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
    # Also save the statistics table as CSV
    pd.DataFrame(results, columns=headers).to_csv("./data/stats.csv", index=False)
    print("\n✅ Saved detailed statistics to stats.csv")


def normalize_udp(series):
    """Same normalization used in activity.py update_belief_optimized"""
    min_v = series.min()
    max_v = series.max()
    return (series - min_v) / (max_v - min_v + 1e-9)


def draw_sample_graphs(df, udp_file):
    udp_df = pd.read_csv(udp_file, sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()

    sampled_pathways = list(df['pathway'].unique())[:3]
    print(f"\n🎨 Drawing 3 pathway")

    for pathway in sampled_pathways:
        subdf = df[df['pathway'] == pathway]
        G = build_graph(subdf)

        # Pick one random sample column from UDP data
        sample_name = udp_df.columns[0]
        sample_udp = to_prob_power(normalize_udp(udp_df[sample_name]))

        # Compute edge weights from UDP values of connecting genes
        weights = []
        for u, v, edge_data in G.edges(data=True):
            gene = edge_data['gene']
            udp_val = sample_udp.get(gene, 0.0)
            G[u][v][0]['weight'] = udp_val
            weights.append(udp_val)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5)

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            width=[2 + 4 * w for w in weights],  # thicker = higher UDP
            edge_color='gray'
        )
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Edge labels (UDP weights)
        edge_labels = {(u, v): f"{G[u][v][0]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.title(f"{pathway} (Sample: {sample_name})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()