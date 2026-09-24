# Business Entity Resolution Challenge: Final Build Plan (v2, 2026-09-25)

Single source of truth for the pipeline. v1 was the original build plan; v2 folds in the Phase 0 numbers from `EXPERIMENTS.md`. Where v2 differs from v1, the section says so and cites the number. Phases stay gated: do not start a phase until the previous phase's diagnostic is logged.

## 0. Problem, metric, hard rules (unchanged)

Three sources of business records (name, address, country). S1 is deduplicated. For every S1 entity output the set of S2/S3 records for the same business: zero, one, or many.

Macro F0.5 per S1 entity, singletons included: empty-vs-empty scores 1.0, predicted-empty-with-true-matches 0.0, any prediction on a singleton 0.0. Precision weighs 2x recall. With three true matches (the typical entity), missing one costs 0.09 on that entity; adding one wrong link costs 0.21. Decisions are made per entity, never per pair.

Non-negotiable rules:

1. All files are TSV. Always read with `sep="\t"`. Never write CSV.
2. No external lookups of any kind. Hand-written normalization dictionaries are fine. Dictionaries learned from the provided training data are fine (see 3.3); nothing from the internet.
3. Any pretrained model must be MIT or Apache 2.0 and at most 8B parameters. Record license and parameter count in `MODELS.md`.
4. Country is an open set. Never one-hot it, never filter on specific values, never branch on `country in {US, India}`. Equality between two records' country strings is the only allowed use. Every test S1 entity, France included, gets a row.
5. `candidate_pairs.tsv` is the exact set the final matcher scores. Every ID in `matching_results.tsv` must be in it. Both files are written from the same in-memory object.
6. Run `utils/validate_submission.py` before every leaderboard upload.
7. Never tune anything on test data. Test is read only by `infer.py`.
8. Log every run in `EXPERIMENTS.md` with config, OOF macro F0.5, macro precision, macro recall, singleton accuracy, multi-match F0.5, and one line of interpretation. No result without a number. Every component after the baseline must beat the previous best OOF or it is removed.

## 1. What Phase 0 established (drives everything below)

| fact | number | consequence |
|---|---|---|
| singletons | 5.6% of S1; 72% of S1 have 3+ matches, mean 3.46 | recall on multi-match entities dominates the score; singleton gate is secondary |
| per-source cardinality | 51% of S1 have 2+ S2 matches, 55% have 2+ S3 | multi-candidate decision rule, no top-1-per-source |
| reverse uniqueness | every matched S2/S3 ID belongs to exactly one S1 | capacity-1 assignment is valid |
| country agreement | 100% of true pairs | partition all retrieval by country equality |
| blocking ceiling | 99.99% of true pairs share a name token OR an address token; name-only keys cap at 72 to 80% on India | retrieval must use name and address together |
| name collisions | 49.5% of S1 share their suffix-free name with another same-country S1 | address decides; hard negatives are same-name entities |
| near-miss structure | same name, same city, different street number; whole-address token_set_ratio still 70 to 90 | numeric-token conflict and street-level similarity are first-class features |
| Indic names | 7.2% of pairs; 1,463 Indic tokens map to English with 99% consistency; 93.5% test coverage | learn a token dictionary from train pairs (decision 3.3) |
| domain names | 3.5% of pairs; 68% equal the concatenated suffix-free name | dedicated feature |
| opaque aliases | 1.8% of pairs; address is copied almost verbatim | matcher must be able to fire on address alone |
| unrecoverable tail | < 0.1% of pairs have both name and address similarity below 50 | recall ceiling above 99% |
| postal codes | 0% India, 11% US | no postal-code blocking; postal features are minor |
| test shift | implied 40% distractor S2/S3 records in test vs 27% in train | validation must inject distractors; thresholds are conservative |
| France | 3 regions, ~20 cities, `r`/`r.`/`n°` abbreviations, first-character-dropped words | char n-grams and street-number features carry France; city tokens do not |
| tokenization | `[^\p{L}\p{N}]+` shatters Indic words | tokens must keep `\p{M}` |

Machine: 12 CPU, 24 GB RAM, GTX 1650 4 GB. Everything below is sized for that. Data: 2.2M/5.0M/5.3M train, 1.7M/4.9M/5.1M test.

## 2. Repository layout (unchanged) and run contract

```
code/business_entity_resolution/
  src/
    io.py            load/save TSV, ID parsing, Parquet caching
    normalize.py     multi-view representations + dictionaries (hand-written and learned)
    translit.py      Indic token dictionary: build from train pairs, apply
    blocking/
      tfidf_char.py  retriever 1 and 2 (char n-gram, name / name+address)
      tfidf_word.py  retriever 3 (word level, name tokens + address alpha tokens)
      dense.py       retriever 4 behind --dense flag (optional)
      union.py       union, cheap post-filter, recall@K diagnostics, writes candidate table
    features.py      pair and entity features
    split.py         grouped folds with distractor injection
    metrics.py       exact macro F0.5, recall@K, collision rate
    train.py         LightGBM 5-fold OOF
    decide.py        entity rule, capacity-1 assignment, ambiguity veto
    infer.py         end-to-end test run, writes output/
    validate.py      wraps utils/validate_submission.py
  README.md          one command per stage; full reproduction from raw TSV
  requirements.txt   pinned
output/              matching_results.tsv, candidate_pairs.tsv
Documentation_template.md, EXPERIMENTS.md, MODELS.md, PLAN.md
```

Every stage caches its output as Parquet under `work/` and can be rerun alone. All seeds fixed at 42. Polars for all bulk table work (pandas only where a library demands it).

## 3. Phase 1: multi-view normalization

Raw fields are never overwritten. Each record carries, side by side:

Name views: `name_raw`, `name_lower`, `name_ascii` (NFKD fold, accents stripped, Indic tokens replaced via 3.3), `name_alnum`, `name_nosuffix`, `name_suffix` (canonical suffix or none), `name_tokens`, `name_first_token`, `name_concat` (suffix-free tokens joined with no spaces, for domain-name matching), `name_is_domain`, `name_domain_stem`.

Address views: `addr_raw`, `addr_norm` (lowercased, unicode-folded, abbreviations expanded, null tokens removed, Indic state and city tokens replaced via 3.3), `addr_tokens`, `addr_alpha_tokens`, `addr_rare_tokens` (alpha tokens whose document frequency in the same-country S2+S3 pool is below 20,000: street names, localities, not city or state), `addr_numeric_tokens` (digit runs; internal spaces collapsed; alphanumerics like `576B` yield `576`), `addr_house_number` (first numeric token), `addr_long_numeric` (numeric tokens of length 5+), `addr_postal_candidate`.

### 3.1 Hand-written dictionaries (open set, never learned)

Legal suffixes, canonicalized to one token each: inc/incorporated/inc.; corp/corporation; llc/l.l.c.; ltd/limited/ltd.; co/company; pc/p.c.; llp/l.l.p.; lp; plc; pvt/private/pvt./prа./प्रा.; m/s; enterprises; traders; sarl/s.a.r.l.; sas/s.a.s.; sasu; sa/s.a.; eurl; snc/s.n.c.; sci; ei; ets/établissements; gmbh; bv; sl; srl; ag; oy; ab; "& fils", "& co", "& sons", "and sons". Honorific prefixes stripped for the nosuffix view: dr, mr, mrs, ms, smt, shri, sri, m/s, the.

Address abbreviations, generic: rd/road, st/street, ave/av/av./avenue, blvd/bd/boulevard, dr/drive, ln/lane, ct/court, hwy/highway, pkwy/parkway, cir/circle, pl/place, sq/square, ste/suite, apt/apartment, fl/flr/floor, bldg/building, unit/#, po box; r/r./rue, all/allée, ch/chemin, imp/impasse, rte/route, n°/nº/no/no./number marker (dropped); nagar, marg, chowk, sector/sec, phase, plot, gala, flat, h.no/h no/house no (dropped marker), opp/opposite, nr/near, bhd/behind. US state abbreviations expanded to full names and India state codes expanded (generic, static list). French region and department names kept as-is.

Punctuation: `&` and `and` equivalent; periods inside abbreviations removed; hyphens and slashes become spaces except inside numeric tokens; whitespace collapsed; literal `null`, `<null>`, `n/a`, `none` removed from addresses.

Tokenizer: split on `[^\p{L}\p{M}\p{N}]+` after folding.

### 3.2 Decision on France recall

No France-specific code. France is handled by the generic French suffixes and abbreviations above, char n-gram retrieval, and street-number features. The "first character dropped" corruption is covered by char n-grams and partial ratio; nothing learns from France.

### 3.3 Decision: learned Indic token dictionary (deviation from v1, adopted)

v1 said dictionaries are hand-written. The number: 23% of India/S2 positives have zero name similarity because of Indic script, and 1,463 Indic tokens explain them with 99% consistency. `translit.py` builds, from training pairs only: for every true pair whose S2/S3 name contains Indic script and has the same token count as the Latin S1 name, align tokens positionally and count (indic_token -> latin_token); keep the majority mapping when it has at least 3 observations and at least 80% share. Do the same for address tokens using co-occurrence (Indic token in addr2 -> most frequent S1 address token that appears with it, restricted to tokens with document frequency above 1,000 so it captures state and city names). Unknown Indic tokens fall back to rule-based ITRANS transliteration (indic-transliteration, MIT). The dictionary is a training artifact like the model: it is rebuilt inside `train.py` from the training folds only, saved to `work/translit.json`, and reused by `infer.py`. It uses no external data. Log its size and test coverage in `EXPERIMENTS.md`.

## 4. Phase 2: high-recall union blocking

All retrieval is partitioned by exact country string (equality only). Each partition builds its own indexes; unseen countries such as France work automatically.

Retrievers, each returning `(s1_id, candidate_id, retriever, score, rank)` for top-N per S1 per source, N = 30 initially:

1. Char n-gram (2 to 4) TF-IDF cosine on `name_ascii`.
2. Char n-gram TF-IDF cosine on `name_ascii + " " + addr_norm`.
3. Word TF-IDF cosine on `name_tokens + addr_alpha_tokens` (IDF handles the rare-token effect measured in Phase 0: rare shared tokens at DF <= 20,000 already give 98.8% recall).
4. Optional, behind `--dense`: `intfloat/multilingual-e5-small` (MIT, 118M) sentence embeddings with an exact or IVF inner-product index. Only built if the union of 1 to 3 misses the gate. On this GPU e5-small embeds all 24M records in about 3 hours; bge-m3 would take a day and is not the default.

Implementation: `sparse_dot_topn` (Apache 2.0) for chunked sparse top-K per country partition, or chunked sklearn matrix products if that library is unavailable; both paths documented. Memory target: one country partition's S2+S3 TF-IDF matrix in RAM at a time.

Union and post-filter: union all retrievers; drop candidates whose country differs; drop candidates that share zero name tokens AND zero address tokens with the S1 record (Phase 0: this loses at most 0.01% of true pairs); cap at the K chosen from the curve. The post-filter output is the candidate table, written to Parquet, and it is what `candidate_pairs.tsv` and the matcher both read.

Diagnostic 4 (recall@K): K in {1, 3, 5, 10, 20, 30, 50, 100}, union and per retriever, overall, per country, per source, and split single-match vs multi-match entities. Gate: union recall@50 >= 98% on train. If not met, fix retrieval before anything else. Expect the gate to be met by retrievers 1 to 3 given the 99.99% token-overlap ceiling; if India lags, check the Indic dictionary first.

Diagnostic 5 (collision rate): share of S2/S3 records that appear in the top-K of more than one S1, and the score gap distribution between the top two claimants. High collision confirms assignment (Phase 6 Step 2) is high priority.

Choose K per source at the point where union recall plateaus. Log K, recall@K, mean candidates per S1, and total candidate rows.

## 5. Phase 3: validation framework (deviation from v1, adopted: distractor injection)

Split S1 entities into 5 grouped folds (seed 42). Fold 5 is held out entirely and scored once at the end.

Distractor injection: inside each validation fold, 20% of its S1 entities are removed from the query set while their true S2/S3 records stay in the candidate pool, so the fold's distractor share matches the ~40% inferred for test. Retrieval and scoring for the fold run against this pool. The number that decides thresholds is the injected-fold macro F0.5. Both the plain and injected scores are logged.

`metrics.py` implements the exact per-entity F0.5 with the three empty-set rules and macro-averages; unit test against the worked example (2 correct, 1 wrong, 2 true -> 0.714) and against pure singleton cases. Report for every experiment: macro F0.5, macro precision, macro recall, singleton accuracy, F0.5 restricted to multi-match entities, and per-country F0.5.

Cross-country diagnostic once the matcher exists: train on one country, validate on the other, both directions. A drop larger than 0.03 F0.5 means a feature is country-specific; find and remove it.

## 6. Phase 4: pair features

Computed for every row of the candidate table.

Name: rapidfuzz ratio, partial_ratio, token_sort_ratio, token_set_ratio on `name_ascii` and on `name_nosuffix`; Jaro-Winkler on `name_nosuffix`; longest common substring length; first-token equality; token Jaccard; `name_len_ratio`; `name2_is_single_token`, `name2_is_domain`, `domain_stem_similarity` (partial ratio between the candidate's domain stem and the S1 `name_concat`); `name2_has_nonlatin`, `name_translit_applied`.

Suffix: `suffix_equal`, `suffix_conflict`, `suffix_missing_one_side`.

Address: token Jaccard and token_set_ratio on `addr_norm`; the same on `addr_rare_tokens` only (street and locality level, city and state excluded; this is the near-miss discriminator); partial_ratio on `addr_norm`; `addr_empty_either`; `addr_len_ratio`.

Numeric, first-class: `house_number_equal`, `house_number_prefix_equal` (576 vs 576B), `numeric_token_jaccard`, `long_numeric_exact`, `numeric_conflict` (both sides have numeric tokens and none match), `n_numeric_shared`, `postal_equal`, `postal_prefix3_equal`.

Country: `country_equal` only.

Retrieval: `retrieved_by_<r>` flags, `retriever_count`, each retriever's score and rank, best retriever score.

Within-entity relative: for the best retriever score and for the model score in a second pass: `score - best`, `score / best`, `score - second_best`, entity rank, `n_candidates_above_0.8`, `n_candidates_above_0.9`, number of candidates from the same source.

Reverse-rank across entities: for candidate j, this S1's rank among all S1 entities that retrieved j, and the score gap to the best competing claimant.

Cross-source consistency (feature only, added in Phase 6 Step 3): best similarity between this S2 candidate and the S1's top S3 candidate, and vice versa.

Embedding cosines only if `--dense` is on.

## 7. Phase 5: matcher

LightGBM binary classifier on candidate rows. Training rows: all positives in the fold's candidate table plus every retrieved negative, capped at the 30 highest-retrieval-score negatives per S1 when memory requires (estimate: 2.2M S1 x 37 rows x ~45 float32 features = 15 GB, so the cap or a 50% entity subsample is expected; log which). Never sample random negatives. Weight negatives retrieved by 2+ retrievers 1.5x. 5-fold OOF probabilities for every row; all downstream tuning uses OOF only. Early stopping on fold logloss; fixed seed; log parameters.

Feature importance is logged; any feature tied to a specific country value is removed.

## 8. Phase 6: entity-level decision layer

All tuning on injected-fold OOF scores against macro F0.5.

Step 1, entity rule: s1 = best candidate score. Predict the best candidate if s1 > tau1. Predict any other candidate j if sj > tau2 and s1 - sj < delta. Grid: tau1 in 0.50 to 0.95, tau2 in 0.30 to 0.90, delta in 0.05 to 0.60. Tune per source and shared; keep whichever wins. This is the baseline.

Step 2, capacity-1 assignment (valid by diagnostic 3): sort all surviving edges by score, claim each S2/S3 record for the first S1 that reaches it, drop later claimants; ambiguity veto: if the runner-up claimant is within margin m of the winner, drop the edge for both. Tune m in 0.02 to 0.30. Keep if it beats Step 1.

Step 3, cross-source consistency feature: retrain with the feature from Phase 4; keep if it beats Step 2.

Step 4, singleton gate: LightGBM on entity summaries (best, second best, gap, best name sim, best address sim, retriever agreement, top S2 score, top S3 score, counts above thresholds). Keep only if it beats Steps 1 to 3.

Step 5, conservatism check: report the chosen thresholds' F0.5 on the injected fold and on a fold with 30% injection. If the score drops more than 0.02 between 20% and 30% injection, raise tau1 until the drop is under 0.01 and log both.

## 9. Phase 7: optional reranker (expected to be skipped)

A 7B LLM does not fit this GPU (4 GB). If Step 5 leaves a large uncertain band (OOF probability 0.3 to 0.8 on more than 5% of candidate rows), try a small permissive cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, Apache 2.0, 22M, fine-tuned on training pairs) on that band only. Keep only if OOF macro F0.5 improves by more than fold variance. Otherwise delete it and say so in `EXPERIMENTS.md`.

## 10. Phase 8: inference and submission

`infer.py`: normalize test, load the training-fold dictionaries and model, block per country partition, write `candidate_pairs.tsv` from the candidate table, featurize, score, decide, write `matching_results.tsv`. One row per test S1, empty for singletons, comma-separated S2/S3 IDs, no quoting, no duplicates. Run `validate.py` (which calls the official validator with `--check-ids`). Fix every issue before upload.

Final package: `output/`, `code/business_entity_resolution/` with README and pinned requirements, filled `Documentation_template.md` (methodology, blocking strategy with measured recall@K, features, model, decision rule, every pretrained model with license and size, the learned Indic dictionary described as a training artifact). Verify a clean environment reproduces both files from the README.

## 11. Order of work and gates

| step | deliverable | gate to proceed |
|---|---|---|
| 1 | `io.py`, `normalize.py`, `translit.py`, `metrics.py` with unit tests | dictionary coverage of train Indic tokens logged; metric test passes |
| 2 | retrievers 1 to 3, `union.py`, diagnostics 4 and 5 | union recall@50 >= 98% on train, overall and per country |
| 3 | `split.py` with distractor injection, candidate tables per fold cached | fold candidate tables written; recall per fold logged |
| 4 | `features.py`, `train.py`, OOF scores | OOF AUC and per-country feature-importance check logged |
| 5 | `decide.py` Steps 1 to 5 | best injected-fold macro F0.5 logged with thresholds |
| 6 | `infer.py`, validator PASS, first leaderboard upload | public score logged next to injected-fold OOF |
| 7 | cross-country diagnostic, held-out fold 5 score, iterate on the biggest error class | every change beats previous best OOF |
| 8 | dense retriever or reranker only if a gate above failed | same rule |
| 9 | package, documentation, clean-environment reproduction | validator PASS from a fresh checkout |

## 12. Working rules for agents (unchanged)

Log every experiment with numbers. Keep raw and normalized fields side by side. When a number disagrees with this plan, the number wins: report it and stop for a decision. No component stays unless it beats the previous best OOF. Keep the dense retriever behind a flag; the sparse path must produce a valid submission alone.
