"""
Benchmark PathBayes on CRC TCGA data comparing against CMSclassifier results.
Classifies colorectal cancer samples into CMS subtypes (CMS1-4) using pathway activities.
Data:
Level 3 TCGA RNA-seq data, downloaded from the TCGA data portal in January 2014.
The data is RSEM-normalized.
After normalization, the data underwent log-transformation (log2 scale).
Non-tumor samples were removed.
Batch effects were corrected using ComBat method.
Additional quality control steps like outlier detection and normalization (including median centering for some data sets) were applied.
"""
import pandas as pd
import numpy as np
from config import *
from sklearn.preprocessing import Normalizer # Unit norm, row wise. # StandardScaler # Normal distribution. MinMaxScaler # [0,1] range, column wise.
from sklearn.decomposition import PCA
from metrics import *
import decoupler
from pathlib import Path


labels = ['MSI-H', 'MSS', 'MSI-L'] #'CMS1', 'CMS2', 'CMS3', 'CMS4'


if __name__ == "__main__":
    #Retrieve gene sets.

    def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
        """
        Parse a gmt file to a decoupler pathway dataframe.
        """
        from itertools import chain, repeat

        pathways = {}

        with Path(pth).open("r") as f:
            for line in f:
                name, _, *genes = line.strip().split("\t")
                pathways[name] = genes

        return pd.DataFrame.from_records(
            chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
            columns=["geneset", "genesymbol"],
        )

    reactome = gmt_to_decoupler("./data/c2.cp.reactome.v7.5.1.symbols.gmt")
    reactome = reactome.rename(columns={'geneset': 'source', 'genesymbol': 'target'})
    print(reactome.columns)
    print(reactome.head())



    """Main benchmark pipeline"""   
    # Load data
    df = pd.read_csv('./data/TCGACRC_expression-merged.zip', sep='\t', index_col=0).T #, nrows=100
    print(df.head())
    labels = pd.read_csv('./data/TCGACRC_clinical-merged.csv')
    # Match samples between expression and labels
    common_samples = list(set(df.index) & set(labels['id']))    
    df = df[df.index.isin(common_samples)]
    labels = labels[labels['id'].isin(common_samples)].copy()
    labels.set_index('id', inplace=True)
    labels = labels.loc[common_samples] # Ensure same order
    labels = labels['microsatelite']
    print(f"data shape after TCGA filter: {df.shape}")

    # decoupler expects the genes as columns.
    # Run GSEA once on all samples
    activity, pvals = decoupler.mt.gsea(
        data=df,
        net=reactome,
        tmin=5,
        verbose=True
    )
    print(activity.head())

    # Cluster requires (n_samples, n_features).
    #print("Clustering (k=3)...")
    # Convert labels to numeric for metrics
    kmeans = clustering(activity, n_clusters=3)
    y_pred_kmeans = kmeans.labels_
    
    # Metrics
    label_map = {label: index for index, label in enumerate(labels)}
    #index_map = {index: label for index, label in enumerate(labels)}
    y_true_numeric = np.array([label_map[label] for label in labels.values])
    silhouette, calinski, special_acc, completeness, homogeneity, adjusted_mi = calc_stats(
        activity, 
        y_true_numeric, 
        y_pred_kmeans,
        debug=True
    )