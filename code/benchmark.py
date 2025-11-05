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
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from config import *
from activity import calc_activity
from udp import udp_main
from sklearn.preprocessing import Normalizer # Unit norm, row wise. # StandardScaler # Normal distribution. MinMaxScaler # [0,1] range, column wise.
from sklearn.decomposition import PCA
from metrics import *


def load_tcga_data(expression_file, labels_file):
    """Load TCGA CRC expression data and CMS labels"""
    
    # Load expression data (genes × samples)
    expr = pd.read_csv(expression_file, sep='\t', index_col=0)
    print(f"Expression data shape: {expr.shape}")
    
    # Load CMS labels
    labels = pd.read_csv(labels_file)
    
    # Filter outliers.
    outliers = pd.read_csv(data_path+'TCGACRC_expression_merged_outlier_strict.txt')
    outliers2 = set(labels.loc[pd.isna(labels["microsatelite"]) | (labels["microsatelite"] == 'Indeterminate')]['id'])
    outset = set(outliers.name).union(outliers2)
    expr = expr.loc[:, ~expr.columns.isin(outset)]
    print(f"Expression data shape after outlier cleanup: {expr.shape}")
    
    # Filter to TCGA samples only
    #tcga_labels = labels[labels['dataset'] == 'tcga'].copy()
    #tcga_labels = tcga_labels[tcga_labels['CMS_final_network_plus_RFclassifier_in_nonconsensus_samples'] != 'NOLBL']
    
    # Match samples between expression and labels
    common_samples = list(set(expr.columns) & set(labels['id']))    
    expr = expr[common_samples]
    labels_matched = labels[labels['id'].isin(common_samples)].copy()
    labels_matched.set_index('id', inplace=True)
    labels_matched = labels_matched.loc[common_samples] # Ensure same order
    print(f"Expression data shape after TCGA filter: {expr.shape} (genes, samples)")

    # Extract labels (final consensus)
    #y = labels_matched['microsatelite'] #CMS_final_network_plus_RFclassifier_in_nonconsensus_samples
    #print(pd.Series(y).value_counts().sort_index())
    
    return expr, labels_matched


def calculate_udp(expr_data):
    """Calculate UDP values for expression data"""
    print("\nCalculating UDP values...")
    
    # PathBayes expects data in specific format
    # Save temporarily for UDP calculation
    expr_data.to_csv('./data/input.csv')
    
    # Calculate UDP using negative binomial (RNA-seq)
    udp_df, _ = udp_main()
    
    # Save UDP
    udp_df.to_csv('./data/output_udp.csv')
    print(f"UDP calculated: {udp_df.shape}")
    
    return udp_df


def select_cms_relevant_pathways(activity_df):
    """
    Select pathways most relevant to CRC CMS classification.
    Based on biological knowledge of CMS subtypes:
    - CMS1: Immune/MSI, hypermutated
    - CMS2: Canonical/epithelial
    - CMS3: Metabolic
    - CMS4: Mesenchymal/stromal
    """
    activity_df = activity_df.T
    # CMS-relevant pathway keywords
    cms_keywords = [
        # Immune (CMS1)
        'immune', 'inflammation', 'cytokine', 'interferon', 'antigen', 'lymphocyte',
        'T cell', 'B cell', 'NK cell', 'chemokine', 'pd1', 'cytotoxic', 'nkc', 'th1', 'tfh', 'th17', 'treg', 'mdsc', 'complement',
        
        # Canonical/WNT (CMS2)
        'wnt', 'signaling', 'cell cycle', 'proliferation', 'egfr', 'erbb', 'mapk', 'pi3k', 'src', 'jak_stat',
        'caspases', 'proteasome', 'cell_cycle', 'translation', 'notch', 'integrin', 'vegf',
        
        # Metabolic (CMS3)
        'metabolism', 'metabolic', 'glycolysis', 'fatty acid', 'citrate', 'sugar', 'glucose', 'fructose',
        'oxidative phosphorylation', 'amino acid', 'sucrose', 'glactose', 'glutomine', 'glutathione', 'nitrogen',
        'tyrosine', 'fatty_acid', 'arachnoid', 'linoleic',
        
        # Mesenchymal (CMS4)
        'tgf', 'beta', 'adhesion', 'ecm', 'extracellular matrix', 'angiogenesis',
        'focal adhesion', 'integrin',
        
        # General cancer pathways
        'cancer', 'p53', 'apoptosis', 'jak-stat', 'mapk', 'pi3k', 'akt',
        'mtor', 'ras', 'nf-kappa', 'hippo', 'notch', 'hedgehog'
    ]
    
    pathway_names = activity_df.index.str.lower()
    
    # Find pathways matching keywords
    selected_mask = pd.Series([False] * len(pathway_names), index=activity_df.index)
    for keyword in cms_keywords:
        selected_mask |= pathway_names.str.contains(keyword, case=False, na=False)
    
    selected_pathways = activity_df[selected_mask]
    
    print(f"\nSelected {len(selected_pathways)} CMS-relevant pathways from {len(activity_df)}")
    print("Sample pathways:")
    for i, p in enumerate(selected_pathways.index[:10]):
        print(f"  {i+1}. {p}")
    
    return selected_pathways.T


labels = ['MSI-H', 'MSS', 'MSI-L'] #'CMS1', 'CMS2', 'CMS3', 'CMS4'


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics matching Table 16 format:
    - Overall accuracy
    - Per-class accuracy (sensitivity)
    - Cohen's Kappa
    """
    # Overall accuracy
    print(f'overall_acc: {accuracy_score(y_true, y_pred)}')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    """Print confusion matrix"""
    print(f"\nconfusion matrix")
    print("-"*50)
    df_cm = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
    print(df_cm)
    print()

    # Per-class sensitivity (recall)
    class_acc = {}
    for i, cms in enumerate(labels):
        if cm[i, :].sum() > 0:
            class_acc[cms] = cm[i, i] / cm[i, :].sum()
        else:
            class_acc[cms] = 0.0
        print(f'class: {cms} acc: {class_acc[cms]}')
    
    # Cohen's Kappa
    print(f'Cohen kappa: {cohen_kappa_score(y_true, y_pred):.2f}')
    
    return cm


if __name__ == "__main__":
    """Main benchmark pipeline"""
    
    print("PathBayes CRC TCGA Benchmark")
    print("="*80)
    
    # 1. Load data
    expr_data, y_true = load_tcga_data(
        expression_file=data_path+'TCGACRC_expression-merged.zip',
        labels_file=data_path+'TCGACRC_clinical-merged.csv'
    )
    
    # 2. Calculate UDP
    #udp_df = calculate_udp(expr_data)
    
    # 3. Calculate activity
    #activity = calc_activity()
    activity = pd.read_csv(data_path+'output_activity.csv', index_col=0) # (samples (rows) × pathways (columns)) activity.shape = 472 x 314
    common_samples = list(set(activity.index) & set(y_true.index)) 
    y_true = y_true[y_true.index.isin(common_samples)]
    activity = activity[activity.index.isin(common_samples)]
    y_true = y_true['microsatelite']
    # 4. Select CMS-relevant pathways (optional - comment out to use all pathways)
    activity = select_cms_relevant_pathways(activity)

    # 4. Scale the data.
    scaler = Normalizer()
    activity = scaler.fit_transform(activity)
    
    # 5. Dimensionality reduction with PCA
    #pca = PCA(n_components=30, svd_solver='arpack')
    #activity = pca.fit_transform(activity)
    #print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
       
    # 6. Cluster with KMeans (4 clusters for CMS1-4)
    print("\nClustering with KMeans (k=3)...")
    # Convert labels to numeric for metrics
    label_map = {label: index for index, label in enumerate(labels)}
    index_map = {index: label for index, label in enumerate(labels)}
    y_true_numeric = np.array([label_map[label] for label in y_true.values])
    kmeans = cluster_with_kmeans(activity, n_clusters=3)
    y_pred_kmeans = kmeans.labels_
    print(f"KMeans clustering complete. Predicted clusters: {len(y_pred_kmeans)}, values: {np.unique(y_pred_kmeans)}")
    
    # 7. Calculate clustering metrics
    print("\n" + "="*80)
    print("CLUSTERING METRICS (Unsupervised)")
    print("="*80)
    
    silhouette, calinski, special_acc, completeness, homogeneity, adjusted_mi = calc_stats(
        activity, 
        y_true_numeric, 
        y_pred_kmeans,
        debug=True
    )
    
    # 8. Convert cluster labels to CMS labels for confusion matrix
    # Map cluster assignments to labels (need to find best matching)
    y_pred_cms = np.array([index_map[label] for label in y_pred_kmeans])
    
    # Calculate metrics
    cm = calculate_metrics(y_true, y_pred_cms)