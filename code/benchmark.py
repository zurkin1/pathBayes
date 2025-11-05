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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from config import *
from activity import calc_activity
from udp import udp_main
from sklearn.preprocessing import Normalizer # Unit norm, row wise. # StandardScaler # Normal distribution. MinMaxScaler # [0,1] range, column wise.
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from metrics import *


def load_tcga_data(expression_file, labels_file):
    """Load TCGA CRC expression data and CMS labels"""
    
    # Load expression data (genes × samples)
    expr = pd.read_csv(expression_file, sep='\t', index_col=0)
    print(f"Expression data shape: {expr.shape}")
    
    # Filter outliers.
    outliers = pd.read_csv(data_path+'TCGACRC_expression_merged_outlier_strict.txt')
    outset = set(outliers.name)
    expr = expr.loc[:, ~expr.columns.isin(outset)]
    
    # Load CMS labels
    labels = pd.read_csv(labels_file, sep='\t')
    
    # Filter to TCGA samples only
    tcga_labels = labels[labels['dataset'] == 'tcga'].copy()
    tcga_labels = tcga_labels[tcga_labels['CMS_final_network_plus_RFclassifier_in_nonconsensus_samples'] != 'NOLBL']
    
    # Match samples between expression and labels
    common_samples = list(set(expr.columns) & set(tcga_labels['sample']))
    print(f"Common samples: {len(common_samples)}")
    
    expr_matched = expr[common_samples]
    labels_matched = tcga_labels[tcga_labels['sample'].isin(common_samples)].copy()
    labels_matched.set_index('sample', inplace=True)
    labels_matched = labels_matched.loc[common_samples] # Ensure same order
    
    # Extract CMS labels (final consensus)
    y = labels_matched['CMS_final_network_plus_RFclassifier_in_nonconsensus_samples'].values
    
    print(f"Final dataset: {expr_matched.shape[1]} samples × {expr_matched.shape[0]} genes")
    print(f"CMS distribution:")
    print(pd.Series(y).value_counts().sort_index())
    
    return expr_matched, y


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


def calculate_metrics(y_true, y_pred, dataset_name='PathBayes'):
    """
    Calculate performance metrics matching Table 16 format:
    - Overall accuracy
    - Per-class accuracy (sensitivity)
    - Cohen's Kappa
    """
    
    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['CMS1', 'CMS2', 'CMS3', 'CMS4'])
    """Print confusion matrix"""
    print(f"\nconfusion matrix")
    print("-"*50)
    df_cm = pd.DataFrame(
        cm,
        index=['CMS1', 'CMS2', 'CMS3', 'CMS4'],
        columns=['CMS1', 'CMS2', 'CMS3', 'CMS4']
    )
    print(df_cm)
    print()

    # Per-class sensitivity (recall)
    class_acc = {}
    for i, cms in enumerate(['CMS1', 'CMS2', 'CMS3', 'CMS4']):
        if cm[i, :].sum() > 0:
            class_acc[cms] = cm[i, i] / cm[i, :].sum()
        else:
            class_acc[cms] = 0.0
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Format results similar to Table 16
    results = {
        'Dataset': dataset_name,
        'Overall': f"{overall_acc:.2f}",
        'CMS1': f"{class_acc['CMS1']:.2f}",
        'CMS2': f"{class_acc['CMS2']:.2f}",
        'CMS3': f"{class_acc['CMS3']:.2f}",
        'CMS4': f"{class_acc['CMS4']:.2f}",
        'Kappa': f"{kappa:.2f}"
    }
    
    return results, cm


if __name__ == "__main__":
    """Main benchmark pipeline"""
    
    print("PathBayes CRC TCGA Benchmark")
    print("="*80)
    
    # 1. Load data
    expr_data, y_true = load_tcga_data(
        expression_file=data_path+'TCGACRC_expression-merged.zip',
        labels_file=data_path+'cms_labels_public_all.txt'
    )
    
    # 2. Calculate UDP
    expr_data = expr_data.round(3)
    #udp_df = calculate_udp(expr_data)
    
    # 3. Calculate activity
    #activity = calc_activity()
    activity = pd.read_csv(data_path+'output_activity.csv', index_col=0) # (samples (rows) × pathways (columns)) 472 x 314

    # 4. Select CMS-relevant pathways (optional - comment out to use all pathways)
    activity = select_cms_relevant_pathways(activity)

    # 4. Scale the data.
    #scaler = Normalizer()
    #activity = scaler.fit_transform(activity)
    
    # 5. Dimensionality reduction with PCA
    #pca = PCA(n_components=30, svd_solver='arpack')
    #activity = pca.fit_transform(activity)
    #print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
       
    # 6. Cluster with KMeans (4 clusters for CMS1-4)
    print("\nClustering with KMeans (k=4)...")
    # Convert CMS labels to numeric for metrics
    cms_to_num = {'CMS1': 0, 'CMS2': 1, 'CMS3': 2, 'CMS4': 3}
    y_true_numeric = np.array([cms_to_num[label] for label in y_true])
    kmeans = cluster_with_kmeans(activity, n_clusters=4)
    y_pred_kmeans = kmeans.labels_
    print(f"KMeans clustering complete. Predicted clusters: {np.unique(y_pred_kmeans)}")
    
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
    # Map cluster assignments to CMS labels (need to find best matching)
    num_to_cms = {0: 'CMS1', 1: 'CMS2', 2: 'CMS3', 3: 'CMS4'}
    y_pred_cms = np.array([num_to_cms[label] for label in y_pred_kmeans])
    
    # Calculate metrics in CMSclassifier format
    results_clustering, cm_clustering = calculate_metrics(
        y_true, 
        y_pred_cms, 
        dataset_name='PathBayes (KMeans)'
    )
    
    print("\n" + "="*80)
    print("CLASSIFICATION METRICS (Table 16 Format - Using KMeans Clusters)")
    print("="*80)
    print(f"{'Dataset':<25} {'Overall':<10} {'CMS1':<8} {'CMS2':<8} {'CMS3':<8} {'CMS4':<8} {'Kappa':<8}")
    print("-"*80)
    print(f"{results_clustering['Dataset']:<25} {results_clustering['Overall']:<10} "
          f"{results_clustering['CMS1']:<8} {results_clustering['CMS2']:<8} "
          f"{results_clustering['CMS3']:<8} {results_clustering['CMS4']:<8} "
          f"{results_clustering['Kappa']:<8}")