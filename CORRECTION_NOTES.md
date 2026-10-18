# Correction patch and corrected runs (added 2026-10-19)

## Diagnosis
The released entity-pair graph aggregation (graph_hts in model_optimized.py) reads node
features from the head of the batch tensor and writes outputs back to the same head for
every document, without per-document start/end offsets. Documents within a batch thus
contaminate one another's graph representations. Combined with the positive-first ordering
of feature construction, this inflates development-set predictions (10,076 vs 8,478
predicted triples on the dev set), while official-test predictions (enumeration order)
are not inflated -- the 9.5-point dev-test gap reported in the paper.

## Patch
model_optimized.py carries an environment-guarded correction: run with FIX_GRAPH=1 to use
correct per-document slicing. Without the variable, behavior is identical to the released code.
prepro.py carries a diagnostic flag SORT_HTS=1 (sorts a document's entity pairs into
enumeration order at feature construction time; used only for the diagnosis experiments).

## Validation
With FIX_GRAPH=1, predictions are order-invariant: re-sorting dev entity pairs yields the
same prediction set (set difference zero).

## Corrected results (five seeds)
- Backbone: dev F1 61.03 +/- 0.22 (IgnF1 59.22)
- Full model: dev F1 61.01 +/- 0.06 (IgnF1 59.16)
- Official test (server display): 0.61 for both
- Per-seed corrected logs (incl. corrected encoder runs): logs_corrected/
- Learned per-pair thresholds of the corrected full model: tau_stats_fixed_full.csv

## Reproducing
    FIX_GRAPH=1 bash retrain_fixed.sh
    FIX_GRAPH=1 bash retrain_encoder_fixed.sh

Trained checkpoints are available from the authors upon reasonable request.
