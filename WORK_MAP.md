# Work map: where everything lives

Server repo root: `/mnt/vaultb/Soham/MLCSoham`. All paths below are relative to it. Run code as `uv run python -m plan1.<module>`.
The running log of every run, finding and leaderboard score is `EXPERIMENTS.md`.

## 1. Submissions (`output/plan1/<name>/matching_results.tsv`; main runs also have `candidate_pairs.tsv`)
| Folder | What | Leaderboard |
| --- | --- | ---: |
| `p1-v3-s2ce/` | **best**: features v3 + twin features + MiniLM cross-encoder | **0.975228** |
| `p1-v3-s2ce-nofrance/` | same, France S1 empty (probe) | 0.845536 |
| `p1-v3-s2ce-noce/` | same pipeline without the cross-encoder | not uploaded |
| `p1-v3-s2/` | older stage 2 (features v2) | 0.952161 |
| `p1-v3-s2-nofrance/`, `p1-v3-s2-usin90/` | probes | 0.830271, 0.953304 |

Rows are in `test_source1.tsv` order; the portal rejects anything else. Write with `plan1/dataio.write_id_lists`.
Per-country score from a probe: France ≈ (main − nofrance) / 0.1498 + 0.056.

## 2. Best run artifacts: `work/plan1/p1-v3/stage2_ce/`
| File | Contents |
| --- | --- |
| `stage2ce_meta.json` | thresholds (cross-encoder model **0.736**, no-cross-encoder model 0.6795), feature lists, panel report |
| `test_scores/*.parquet` | **test pairs scored**: `s1_id, target_id, source, p1, ce, p` (cross-encoder model), `p_noce`. Only kept pairs (p1 ≥ 1e-3, top 30 per S1); every other pair counts as rejected |
| `stage1_full.xgb.json` | first-stage XGBoost (features v3, 94 features, 5,936 trees) |
| `stage2_ce.xgb.json`, `stage2_noce.xgb.json` | second-stage XGBoost models |
| `cross_encoder/` | fine-tuned cross-encoder, Hugging Face format (paraphrase-multilingual-MiniLM-L12-v2, Apache 2.0); `ce_meta.json` holds its held-out AUC 0.9971 |
| `base_{fit,gbm_extra,tune,c_select,dev,fresh}.parquet` | labelled kept pairs with all stage-1 + stage-2 features, `label`, `p1` |
| `ce_{set}.parquet` | cross-encoder logit per kept pair: `s1_id, target_id, ce` |
| `ce_pairs_ce_train.parquet` | cross-encoder training pairs: `s1_id, target_id, source, label, p1` |
| `pred_{ce,noce}_{c_select,dev,fresh}.parquet` | panel predictions with labels: `s1_id, target_id, label, p` |
| `stage1_features_{set}_f3.parquet` | full first-stage feature tables (all candidates, labelled) |
| `p1_fit_oof.parquet` | out-of-fold first-stage scores for Fit |

**Loading the cross-encoder.** `AutoTokenizer` / `AutoModelForSequenceClassification.from_pretrained(".../cross_encoder")`, 1 logit.
- Input text comes from `plan1/cross_encoder.record_text`: `"<name> ; <address>"`. The record side is prefixed `"s2 "` / `"s3 "`.
- Names are lowercase, accents folded, Indian script in English letters, dotted legal forms joined; the address is `addr_norm` from features v3.
- Use `score()` in the same file: sorted by length, bf16, about 13.5k pairs/s on one A6000.

## 3. Shared data (`work/plan1/p1-v3/`, `work/shared/`)
| Path | Contents |
| --- | --- |
| `work/shared/raw/{train,test}_records.parquet` | all three sources in one table: `entity_id, business_name, business_address, country, source` |
| `work/shared/splits/manifest.parquet` | every train S1: `role` (fit/tune/calibrate/audit), `sample` (fit_sample, tune_sample, c_select_sample, audit_panel = "dev"), country |
| `work/shared/splits/audit_panel_2.tsv` | the "fresh" panel (20k S1) |
| `work/plan1/p1-v3/enriched_{train,test}_v4_f3.parquet` | per record: `name_cons, name_red, addr_norm, legal_bits, hn` (house number), `raw_name, raw_addr, name_ntok, name_unknown_share, q_name_s1_rate, t_name_rate, t_addr_rate` |
| `work/plan1/p1-v3/normalized_{train,test}_v4.parquet` | search text (normalize v4) |
| `work/plan1/p1-v3/candidates_train.parquet` | candidates for fit (300k) + tune + c_select + dev + fresh: `s1_id, target_id, source, score, rank, fb_score` |
| `work/plan1/p1-v3/candidates_extra.parquet` | 600k more Fit-role S1, same columns + `part` (ce_train / gbm_extra) |
| `work/plan1/p1-v3/test_chunks/*.parquet` | **all test candidate pairs** (about 100 per S1) |
| `work/plan1/p1-v3/translit.json` | Indian-script → English word dictionary learned from Fit pairs |

## 4. Code (`plan1/`)
| Module | Purpose |
| --- | --- |
| `stage2ce.py` | current pipeline: `stage1`, `train [noce]`, `test` (`AMC_SHARD=i/n` for several GPUs). Holds the **twin features** (`add_twin_features`) and the cross-encoder context features |
| `cross_encoder.py` | cross-encoder text, training (`train`), scoring (`score`) |
| `enrich.py` | features v3 per-record extras: French legal forms as separate tags, French regions/departments and n°/no markers dropped, raw text |
| `features.py` | pair features (`FEATURES`: similarity, legal categories, house-number kind, extra words, frequency, raw-text) |
| `stage2.py` | stage-2 context + sibling features (`add_stage2_features`) |
| `normalize.py` | text cleanup; `word_map(..., leftmost=True)` fixes the "S.A.S.U." → "sas u" bug |
| `retrieve.py`, `retrieve_extra.py` | TF-IDF search per country/source (top 50) + address-less name fallback |
| `variants.py` | leaderboard variants from saved scores (`stage2`, `s2ce`): nofrance, france, usin85/90/95, margin, cap, crowd |
| `density_fix.py` | per-kind copy-density correction (tested, not uploaded) |
| `france_check.py`, `test_vs_dev.py` | diagnostics: French legal-form changes; test vs dev kinds of links; S1 twins |
| `dataio.py`, `decide.py`, `metric.py`, `splits.py` | I/O and submission writer, one-owner rule, F0.5, splits |

## 5. Rebuild the best run
```
AMC_BACKEND=xgboost uv run python -m plan1.stage2ce stage1
CUDA_VISIBLE_DEVICES=1 uv run python -m plan1.cross_encoder train
CUDA_VISIBLE_DEVICES=1 uv run python -m plan1.cross_encoder score
AMC_BACKEND=xgboost uv run python -m plan1.stage2ce train
CUDA_VISIBLE_DEVICES=i AMC_SHARD=i/3 AMC_BACKEND=xgboost uv run python -m plan1.stage2ce test   # i = 0, 1, 2, then once without AMC_SHARD
uv run python -m plan1.variants s2ce
```

## 6. Stitching per country (every link stays inside one country)
- Take US/India rows from one `matching_results.tsv` and France rows from another. The France S1 list is
  `work/shared/raw/test_records.parquet` with `source == "S1"` and `country == "France"`.
- Write with `dataio.write_id_lists(pairs, test_s1_roster(), path, "matched_entity_ids")`, then run
  `utils/validate_submission.py` on it.
