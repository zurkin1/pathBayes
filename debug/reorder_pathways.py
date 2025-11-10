'''
Sort pathway_relations.csv in per pathway topological order to save runtime.
'''
import pandas as pd
import pandas as pd
import networkx as nx
from config import *
from tqdm import tqdm


INPUT_FILE = "./data/pathway_relations.csv"
OUTPUT_FILE = "./data/pathway_relations_toposorted.csv"
MAX_ITER = 1000


def break_cycles(G, max_iterations=MAX_ITER):
    """Remove edges with lowest betweenness until graph becomes acyclic."""
    iteration = 0
    while not nx.is_directed_acyclic_graph(G) and iteration < max_iterations:
        iteration += 1
        try:
            cycle = nx.find_cycle(G, orientation="original")
            cycle_edges = [(u, v, k) for u, v, k, _ in cycle]

            # Compute edge betweenness on simplified graph
            G_simple = nx.DiGraph(G)
            bet = nx.edge_betweenness_centrality(G_simple)
            edge_bet = {(u, v, k): bet.get((u, v), 0) for u, v, k in cycle_edges}

            # Remove edge with minimum betweenness
            min_edge = min(edge_bet.items(), key=lambda x: x[1])[0]
            G.remove_edge(*min_edge)
        except nx.NetworkXNoCycle:
            break

    if iteration >= max_iterations:
        print("⚠️ Warning: could not break all cycles after max iterations.")
    return G


def toposort_pathway(subdf):
    """Build pathway graph → break cycles → return interaction IDs in topo order."""
    G = nx.MultiDiGraph()
    for _, row in subdf.iterrows():
        G.add_node(row.name, source=row["source"], target=row["target"], type=row["interactiontype"])

    # Add edges based on shared genes (target of one → source of another)
    for i1, row1 in subdf.iterrows():
        for i2, row2 in subdf.iterrows():
            if i1 == i2:
                continue
            if set(row1["target"]) & set(row2["source"]):
                G.add_edge(i1, i2)

    G = break_cycles(G)
    if not nx.is_directed_acyclic_graph(G):
        print(f"⚠️ Still cyclic pathway, skipping topo order for {subdf['pathway'].iloc[0]}")
        return subdf

    sorted_nodes = list(nx.topological_sort(G))
    return subdf.loc[sorted_nodes]


def main():
    df = pd.read_csv(INPUT_FILE)
    # Parse gene lists (they are '*' separated strings)
    df["source"] = df["source"].fillna("").astype(str).str.lower().str.split("*")
    df["target"] = df["target"].fillna("").astype(str).str.lower().str.split("*")

    reordered = []
    for pathway, subdf in tqdm(df.groupby("pathway", sort=False)):
        sorted_subdf = toposort_pathway(subdf)
        reordered.append(sorted_subdf)

    df_sorted = pd.concat(reordered)
    df_sorted["source"] = df_sorted["source"].apply(lambda x: "*".join(x) if isinstance(x, list) else x)
    df_sorted["target"] = df_sorted["target"].apply(lambda x: "*".join(x) if isinstance(x, list) else x)
    df_sorted.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved topologically ordered pathways to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()