import pandas as pd
import pandas as pd
import ast


INPUT_FILE = "./data/pathway_relations_toposorted.csv" # input (with ['gene1', 'gene2'] format)
OUTPUT_FILE = "./data/pathway_relations_fixed.csv" # output (with gene1*gene2 format)

def normalize_gene_field(value):
    """Convert stringified list ['a', 'b'] → 'a*b'; keep plain strings unchanged."""
    if pd.isna(value) or value == '':
        return ''
    value = str(value).strip()
    # Detect and safely parse Python-style list strings
    if value.startswith('[') and value.endswith(']'):
        try:
            genes = ast.literal_eval(value)
            if isinstance(genes, list):
                return "*".join(str(g).strip().lower() for g in genes if g)
        except (SyntaxError, ValueError):
            pass
    # Already in star-separated format or single gene
    return value.lower().replace("'", "").replace('"', '').strip()

def main():
    df = pd.read_csv(INPUT_FILE)
    for col in ["source", "target"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_gene_field)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Converted {INPUT_FILE} → {OUTPUT_FILE} in original '*' format")

if __name__ == "__main__":
    main()