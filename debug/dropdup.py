import pandas as pd


# Load the CSV file
df = pd.read_csv("c:/github/PathBayes/code/data/pathway_relations.csv")

# Identify duplicate rows (all columns)
duplicate_rows = df[df.duplicated(keep=False)]

# Print duplicate rows and their count
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")
print("Duplicate rows:")
print(duplicate_rows)

# Drop duplicate rows, keep only the first occurrence
df_cleaned = df.drop_duplicates()

# Save to new CSV file
df_cleaned.to_csv("c:/github/PathBayes/code/data/pathway_relations_nodup.csv", index=False)

'''
(venv) C:\github>C:\github\venv\Scripts\python.exe c:/github/PathBayes/debug/dropdup.py
Number of duplicate rows: 4485
Duplicate rows:
                                                 pathway                                             source                      interactiontype             target
72                                     Adherens junction                                         actb*actg1                  binding/association        actn4*actn1
73                                     Adherens junction                                         actb*actg1                  binding/association                vcl
74                                     Adherens junction                                         actb*actg1                  binding/association                vcl
75                                     Adherens junction                                         actb*actg1                  binding/association        actn4*actn1
84                                     Adherens junction                               ctnna1*ctnna2*ctnna3                  binding/association        actn4*actn1
...                                                  ...                                                ...                                  ...                ...
20353                             VEGF signaling pathway                                        plcg1*plcg2                  compound$activation  prkca*prkcb*prkcg
20376                          Vibrio cholerae infection                                              prkca  compound$activation$phosphorylation        plcg1*plcg2
20379                          Vibrio cholerae infection                                              prkca  compound$activation$phosphorylation        plcg1*plcg2
20442  Viral protein interaction with cytokine and cy...  ccl26*ccl27*ccl28*ccl5*ccl7*ccl11*ccl13*ccl15*...                           activation               ccr3
20444  Viral protein interaction with cytokine and cy...  ccl26*ccl27*ccl28*ccl5*ccl7*ccl11*ccl13*ccl15*...                           activation               ccr3

[4485 rows x 4 columns]
'''