# PathWeigh v2
## Core Modifications to activity.py

## Interaction Aggregation Logic in process_pathway()

Current logic:  
- Sum input genes → scale → accumulate to pathway activity  

New logic:  
- Initialize all genes with UDP values  
- Run LBP (Loopy Belief Propagation) message passing until convergence  
- Extract final beliefs  


## LBP Iteration Loop (Before Returning pathway_activity)
Initialization:
`beliefs = {gene: udp[gene] for gene in pathway}`

Iteration (20–30 times):  
- Update each non-input gene's belief based on incoming messages from parent interactions  
- Check for convergence:  
`max(abs(new_belief - old_belief)) < 1e-3`

## Message Computation per Interaction Replaces interaction_activity
For each interaction `(sources, type, targets)`:

`message = combine_parent_beliefs(sources) * CPT_weight(type)` 

CPT: Conditional Probability Table

CPT Weights:  
- Activation → 0.85  
- Inhibition → 0.15 (invert probability logic)  

## Final Pathway Activity
After LBP convergence:

`pathway_activity = mean([beliefs[gene] for gene in all_targets])`

## Key Data Structure Changes
- Track gene-level beliefs throughout the pathway, not just interaction activities.
- Pathway_relations.csv already encodes the graph structure (sources → targets), ideal for Bayesian Network edges.