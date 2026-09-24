# Final Plan 1 — Minimal first submission

**Build:** basic text normalization → word TF-IDF retrieval → 12 pair features → one LightGBM model → one global threshold.

This is an independent, deliberately small solution for the first functional submission. Plans 2 and 3 run concurrently and own the additional complexity. Abundant compute does not make another modeling mechanism necessary.

**No Indic-to-Latin transliteration, learned dictionaries, character retrieval, website/alias processing, calibration, ensembles or ownership assignment in Plan 1.** An addition justified mainly by an expected gain of around 1% stays outside this plan. Actual marginal gains are unknown until measured; this is a scope decision, not a claim of experimentally proven Pareto optimality.

This document supersedes both earlier Plan 1 drafts, including their inherited shared requirements. It specifies work to implement; it does not report a trained model or an existing submission.

## 1. Why this is the minimum worth building

| Keep | Why it earns its place |
| --- | --- |
| Basic text cleanup | Prevents case, punctuation and legal-form differences from dominating comparisons |
| Word search over name plus address | Supplies a manageable candidate set while retaining address-only evidence |
| A compact learned pair matcher | Combines name, address and numeric evidence; fixed rules left a substantial measured gap |
| One threshold optimized for the actual score | Converts pair scores into zero, one or many matches per business |
| Held-out evaluation and output validation | Establishes whether the system works and produces an acceptable submission |

The historical full-target-pool benchmark recovered **97.46% of true links with word retrieval**, versus **98.36% with the larger word/character union**, on 1,000 S1 queries. The difference is candidate recall, not final F0.5. That benchmark used Unidecode and different normalization/partitioning, so its recall is a reference, **not this plan's measured recall**. [Retrieval evidence](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/RESULTS.md).

On a separate, weaker shortlist, the notebook's boosted-tree matcher scored 0.713 versus 0.563 for its hand rule. That comparison supports keeping a learned matcher. It does not establish the score of this single-model, 12-feature version. [Consolidated findings](C:/Users/idcoc/Downloads/EDA_FINDINGS_ALL.md).

The intended separation is:

| Plan | Distinct modeling scope |
| --- | --- |
| **1** | One word retriever, direct pair evidence, one model and threshold |
| **2** | Better cross-script and typo coverage, richer evidence, hard negatives and stronger set/ownership decisions |
| **3** | A strong direct matcher plus validated record relationships and a justified residual-error specialist |

Plan 1 stops at its first measured, valid result. Additional modeling experiments belong to the other runs.

## 2. Data and the shared evaluation contract

Read the original files under student_resource/dataset with an explicit tab separator. Inputs contain entity_id, business_name, business_address and country; labels map source1_entity_id to matched_entity_ids. Preserve IDs as strings and parse an empty label field as an empty set.

Use a single saved split manifest shared by all three plans. Assign S1 owners by a stable, salted hash, with the existing role convention:

| Role | Fraction of S1 owners | Plan 1 use |
| --- | ---: | --- |
| Fit | 60% | Fit on 100,000 sampled S1 and all their retrieved candidates |
| Tune | 10% | 20,000 sampled S1 for early stopping and development diagnostics |
| Calibrate | 10% | Split equally into C-prob and C-select; choose the threshold on 20,000 C-select S1 |
| Audit | 20% | Score one shared 20,000-S1 comparison panel after freezing the configuration |

C-prob is unused by Plan 1; the other plans may use it for probability calibration. The comparison panel comes from Audit, **not C-prob or C-select**, so no plan trains or tunes on another plan's comparison rows. Keep the remaining Audit owners reserved.

Use independent hash salts for role assignment, the Calibrate subdivision and sampling. Sample proportionally by country and true match-count bands, including zero matches. Save the exact sample IDs once. Do not sort by the role-assignment hash and reuse that ordering for samples or folds.

All labeled pairs for one S1 query follow its role. No held-out query labels enter fitting or feature construction. Every query searches **all training S2/S3 targets of the same country**, including targets outside its owner's role; this is a query-held-out evaluation against the full unlabeled target pool. Do not remove distractors, inject missed positives or use known target ownership as a feature.

Score each frozen plan once on the common comparison panel when it is ready; Plan 1 need not wait for the others. After inspecting that panel, do not tune on it and continue calling the result independent. Earlier EDA already used broad data statistics, so call this a prospectively locked comparison, not a historically untouched test.

## 3. Keep normalization small and deterministic

Preserve original fields. Create only a conservative name, a reduced name, a normalized address and its number-token set.

1. **Conservative name:** Unicode normalization and case folding; fold Latin accents; replace ampersands with “and”; convert punctuation to spaces and collapse whitespace. Preserve other scripts and their combining marks. Do not call Unidecode/AnyAscii on Indic text.
2. **Reduced name:** remove a fixed list of whole-token legal forms from the conservative name, wherever they occur. Include inc/incorporated, corp/corporation, co/company, llc, llp, lp, pllc, pc, ltd/limited, pvt/private and SARL/SAS/SASU/EURL/SCI. Remove only the small function-word list and/the/of. Keep words such as services, trading and india. Fall back to the conservative name if reduction empties it.
3. **Address:** normalize actual missing values and documented null placeholders; apply the same basic text cleanup. Use a short, fixed address-only abbreviation table: road→rd, street/saint→st, avenue/ave→av, boulevard→blvd, drive→dr, lane→ln and rue→r. Preserve bis/ter and the remaining address words.
4. **Number tokens:** extract address tokens containing digits before punctuation cleanup. Retain attached letters and internal slashes/hyphens; normalize leading zeros in numeric runs. Thus 002078→2078, while 12B, 12 and 12/3 remain distinct. Do not attempt to identify house, unit, floor or postal-code types.

Word tokenization must preserve Unicode letters, marks and digits, equivalent to [\p{L}\p{M}\p{N}]+. Accent folding must not erase Indic vowel signs.

There is **no** state-alias catalogue, learned normalization, Indic legal-form table, website-stem/joined-name view, trade-name parser, OCR repair, rare-token view or address parser. Those are Plan 2 experiments.

Country is an open string label used for equal-country retrieval partitions. Process every observed country, including France, through the generic path. Do not encode country as a fixed US/India model feature.

## 4. Use one retrieval channel

For both S1 and targets:

    retrieval_text = reduced_name + " " + normalized_address

Build one word-unigram TF-IDF index per country and target source. Initial fixed settings:

    min_df = 2
    max_df = 1.0
    max_features = 300000
    sublinear_tf = True
    smooth_idf = True
    norm = l2
    dtype = float32

Retrieve top 50 S2 and top 50 S3 candidates by cosine score: **at most 100 candidates per S1**. Keep positive-score results; do not pad short or empty lists with arbitrary zero-score targets. Rank within source and break ties by target ID.

Use the [existing sparse top-K benchmark](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/benchmark.py) as the implementation reference. Keep indexes loaded, process query batches and never allocate a dense query-by-target matrix. No exact-name, shared-name-word or number-agreement filter follows retrieval.

Measure retrieval recall and warm-batch throughput on Tune before full inference. A genuine implementation defect must be fixed; a known Indic, typo or missing-address weakness is reported for the other plans. Do not silently add another retriever to pass an arbitrary recall gate.

For inference, fit vocabulary/IDF on the same-country test targets using the same configuration. This uses unlabeled target text, not test labels. All supervised model parameters remain frozen. Retain an S1 roster so queries with no candidates still receive output rows.

## 5. Train one matcher with exactly 12 features

Use RapidFuzz string comparisons and simple set arithmetic. Similarities are scaled to 0–1.

| ID | Feature | Purpose |
| --- | --- | --- |
| 01 | Reduced-name string ratio | Overall spelling agreement |
| 02 | Reduced-name token-set ratio | Word reordering and extra legal/name words |
| 03 | Conservative-name string ratio | Evidence that aggressive reduction could remove |
| 04 | Address token-set ratio | Agreement despite reordered or omitted components |
| 05 | Address partial-string ratio | Shortened address overlap |
| 06 | Address length ratio: shorter/longer | Distinguishes substantial agreement from a tiny contained fragment |
| 07 | Candidate address missing | Separates absent evidence from disagreement |
| 08 | Number-token Jaccard | Shared numeric evidence |
| 09 | Both number sets present but disjoint | Explicit numeric conflict |
| 10 | Word retrieval cosine score | Weighted joint name/address overlap |
| 11 | Retrieval rank within source | Relative retrieval strength |
| 12 | Source indicator, S2 = 0 and S3 = 1 | Different source noise patterns |

Undefined similarities, empty-set Jaccard and undefined length ratios become missing values, not perfect agreement. Numeric conflict is 1 only when both number sets are nonempty and disjoint; otherwise 0. A different number is evidence, never a hard rejection.

Use every retrieved candidate for the 100,000 Fit queries. Label a pair positive exactly when its target is in that query's ground truth. The retrieval pool naturally supplies difficult negatives; do not replace it with easy random negatives.

This gives at most ten million training pairs. Twelve float32 values per pair occupy at most 0.48 GB before IDs, text, LightGBM structures and temporary allocations.

Fit **one LightGBM binary classifier** with:

    learning_rate = 0.05
    num_leaves = 63
    max_depth = 10
    min_data_in_leaf = 100
    lambda_l2 = 5
    max_rounds = 2000
    early_stopping_rounds = 100
    validation_metric = binary_logloss
    seed = 42

Use Tune for early stopping. Pin the library version, thread count and deterministic settings. Keep default unweighted training: no class reweighting, negative mining, cross-validation, model ensemble, parameter sweep or probability calibrator.

Do not refit on Tune/C-select/Audit before test inference; submit the evaluated model. Enlarging the training population is a separate experiment, not a prerequisite for this first submission.

## 6. Select one threshold using macro F0.5

For each S1, retain **every** candidate whose model score is at least one global threshold. Empty predictions and multiple matches from either source are allowed.

For true count g, predicted count k and correct count t:

    F0.5 = 1.25 × t / (0.25 × g + k)

Empty truth plus empty prediction scores 1; all other zero-correct cases score 0. Average over **every S1 query**, including those with no retrieved candidates. Unretrieved true matches remain in g.

Choose the threshold on C-select, using score breakpoints or a coarse sweep followed by local refinement. Include the empty-all option. On an exact objective tie, choose the more conservative threshold. Early stopping and threshold selection are separate from the Audit comparison.

A single raw-score threshold does not need calibrated probabilities: a shared monotone calibration map cannot improve the available ranking-based threshold sets. Do not add isotonic calibration or per-source/country thresholds.

Report the following on Tune/C-select and the frozen comparison panel, clearly separating development from comparison results:

- Macro F0.5 and link precision/recall, with population counts.
- Candidate recall and candidate-oracle macro F0.5.
- US/India results, Indic-name and missing-address retrieval recall, and singleton false-positive rate.
- Mean candidates/matches per S1, empty-answer rate and targets accepted by multiple S1.
- Measured search, feature, training and inference costs.

The Indic flag is for diagnostics only. No labeled France score is available. Check French parsing and coverage, but do not force French match rates to resemble US/India.

There is no cross-S1 ownership resolution. Record collisions as an accuracy diagnostic. The official output contract prohibits duplicates within a list; it does not make reuse of a target across S1 rows a file-format failure.

## 7. Run independently and finish the first submission

Use separate run directories:

    work/plan1/p1-baseline-v1/
    output/plan1/p1-baseline-v1/
        candidate_pairs.tsv
        matching_results.tsv

Plans 2 and 3 use their own namespaces. Share raw input caches and immutable split/metric definitions where useful. Normalized caches, indexes, models, thresholds and outputs belong to their own configurations. Do not overwrite another plan's artifacts or delete the existing benchmark caches.

| Step | Deliverable and completion check |
| --- | --- |
| 1 | Saved inputs/splits and a small fixture; confirm metric edge cases, string IDs and TSV headers |
| 2 | Normalizer and word retrieval; confirm Unicode/number preservation, real same-country candidates and measured recall |
| 3 | Feature builder and one model; confirm labels, missing-value handling and no ID/ownership leakage |
| 4 | One selected threshold, frozen configuration and common-panel report; record actual performance |
| 5 | Full test inference in checkpointed batches; export every S1, including France and empty answers |
| 6 | Both validated outputs, runnable code and saved model/configuration; first leaderboard file ready |

Before full inference, confirm the learned pipeline improves on predicting empty for every query and is not obviously broken relative to the notebook's simple name/address rule. Diagnose a failure; do not turn this into a multi-model search. There is no promised minimum leaderboard score or fixed runtime gate.

Stream inference features rather than storing all test feature rows. Save input/configuration hashes, sample IDs, feature order, model, threshold, pinned dependencies and exact reproduction commands. A small uncached rerun must reproduce saved candidates and predictions.

The two TSV schemas are:

    matching_results.tsv: source1_entity_id<TAB>matched_entity_ids
    candidate_pairs.tsv:  source1_entity_id<TAB>candidate_entity_ids

Write one row per test S1, comma-separated target IDs, and empty fields for empty lists. Require existing same-country S2/S3 IDs, no duplicates within lists, and final matches contained in candidates. The candidate file must contain exactly the records the matcher actually scored.

Run from the repository root:

    python student_resource/utils/validate_submission.py --matching output/plan1/p1-baseline-v1/matching_results.tsv --candidate output/plan1/p1-baseline-v1/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids

Also enforce candidate-file existence and match-subset membership internally: the official validator can skip a missing candidate file and only warns on subset violations.

Only matching_results.tsv is uploaded for the leaderboard. Keep both files for the final package, staged under that package's output/ directory. Follow the [official package specification](D:/PROJECTS/amazon_ml_challenge/student_resource/README.md) for runnable source, pinned requirements and the methodology document. LightGBM is the selected MIT-licensed model; no external record lookup or hosted conversion is involved.

**Stop:** once the pipeline produces a measured, validated first submission, freeze it. Record the portal's actual score after upload. Further quality improvements continue in the other plans rather than expanding Plan 1.

## 8. What changed after critically reviewing both drafts

| Issue | Decision |
| --- | --- |
| My initial plan included two retrievers, a dictionary, ownership handling and extensive ablations; the first revision still kept a dictionary and 24 features | Both were too broad for this milestone. Final scope is one word retriever and 12 direct features, with no transliteration or specialist preprocessing |
| Fable describes “one model” but specifies five fold models plus isotonic calibration | Use one model and optimize the raw-score threshold |
| Fable's “34 features” enumeration contains 39, including unnecessary single-channel rank derivatives | Use the explicit 12-feature catalogue; no RRF, constant retriever-count or feature expansion |
| Fable samples the lowest role-hash values and assigns folds from that same ordering | Its sample concentrates in the first folds, leaving later folds empty. Use independent sampling hashes; this plan has no folds |
| Fable learns a dictionary before its out-of-fold procedure and rebuilds it on all labels for test | That does not validate the entire deployed pipeline out of fold. Dictionaries are removed here; later plans must validate their learned preprocessing and deployed artifacts |
| Website/joined-name handling, state catalogues and typed numbers were treated as necessary | Defer them. Their incremental end-to-end gain has not been established |
| AUC ≥ 0.98, feature importance below 40%, exact-text quotas and France rates within ±25% were release gates | Remove them. They do not establish valid or useful macro set matching |
| Holdout discrepancies triggered further threshold changes | Freeze before comparison; any subsequent tuning makes that panel development data |
| Indic dictionary coverage and an old oracle ratio supported a 0.88–0.92 score forecast | Coverage is not matching gain, and the oracle ratio does not transfer across pipelines. Report measurements |
| Earlier plans assumed sequential upgrades | Run all three independently, with the same comparison panel and isolated artifacts |

Review sources: [Fable Plan 1](D:/PROJECTS/amazon_ml_challenge/plans/fable/PLAN_1_MINIMAL.md), [its shared specification](D:/PROJECTS/amazon_ml_challenge/plans/fable/00_COMMON.md), and the consolidated findings linked above. Their proposed instructions and gates were evaluated as proposals, not adopted as requirements.
