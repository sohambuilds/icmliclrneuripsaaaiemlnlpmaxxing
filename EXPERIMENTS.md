# Experiments log

**Goal: 0.985+ on the test leaderboard.** Top-100 ≈ 0.98, top-300 ≈ 0.97; 0.95 is not worth anything.
Budget: 5 leaderboard submissions per day. The private leaderboard decides — never tune to the public one.

Rules we keep: models MIT/Apache and ≤ 8B; no outside lookups or services on records; learned things (dictionary,
models) from Fit-role training data only; same pipeline for train and test; submission rows in test_source1.tsv order
(portal rejects otherwise, even when the validator passes).

## Runs

| Run | What changed | Dev panel | Fresh panel | Leaderboard |
| --- | --- | ---: | ---: | ---: |
| p1-baseline-v1 | Plan 1: small cleanup, word TF-IDF top 50/source, 12 features, LightGBM, one threshold | 0.9123 | – | 0.8844 |
| p1-baseline-v1 + one owner | each record kept for its best S1 only (not uploaded separately) | – | – | – |
| p1-v2 | Indic dictionary, states in one form, number markers, French/web words, 28 features, 300k Fit, one owner | 0.9523 | – | 0.9388 |
| p1-v3 | + no-address name fallback search, 8000-round cap (stopped 4266), better dictionary, fresh panel | 0.9537 | 0.9531 | 0.9404 |
| p1-v3-crowd10 | + drop records claimed by 10+ S1 | – | – | 0.940413 (no change) |

Leaderboard ≈ panel − 1.3 to 1.4 (test has ~40% unmatched S2/S3 records vs 26% in train, plus France).

## Findings

- **Search is not the problem**: 98.9% of true links found (oracle F0.5 0.997). The decision step loses the rest.
- **Loss split (p1-v3, dev panel)**: rejected true links 3.0 pts, wrong links 0.9, search misses 0.3, no-match
  businesses 0.3.
- **Planted near-copies (traps)**: 62% of wrong links are records that belong to NO business, identical to the S1
  except one change: legal form (LLC→Corp, Inc→Ltd, Pvt Ltd→Public Ltd), one extra word ("Metro", "Harbor",
  "Overseas", "Partners"), or one number (301→30, 3001→3005). Our cleanup deletes legal forms, so the model is blind.
- **Rejected true links**: 35% have no address (1.4% of accepted true do), 30% have number typos (281→481,
  1900→900, 19303→39303) that our exact-number features treat as conflicts, and invented alias names
  ("Gilddrextavo") with a matching address are rejected at p≈0.
- **"Accept records that look like an accepted record" fails**: adds more wrong than right, because copies look alike too.
- **Backends (100k Fit, same data)**: LightGBM 0.9507 (369 s), XGBoost GPU 0.9501 (66 s), CatBoost GPU 0.9487 (170 s).
- Indic dictionary: India 0.880 → 0.947 (v2). Fallback search: no-address recall 86% → 94%, tiny score gain.
- Leak checks: IDs and row order carry no signal.

## Failures / dead ends

- Stricter crowd rule (10+ claimants): no effect on the leaderboard.
- Retraining longer / more rounds: +0.1 pt at most. More Fit data alone: small.
- Generic tuning won't reach 0.98; the traps need features that see *what kind* of difference a pair has.

## Open / next

1. trap_analysis output (legal form, extra words, numbers, no-address, aliases, copy structure) → section below.
2. Build: legal-form / extra-word / number-typo / alias features in both stages; XGBoost GPU; stage 2.
3. If a gap remains: fine-tuned multilingual cross-encoder (MiniLM / XLM-R) on pair text as a stage-2 feature.
4. France probe (France-only submission) to measure France; self-training on test if France is weak.

## Trap analysis results (100k Fit-role S1 + whole train; tables in work/plan1/p1-v3/trap_analysis/)

"Similar" = name token-set ≥ 0.9 and (address ≥ 0.8 or no address). Per S1: 2.7 similar true vs 7.3 similar wrong
candidates. 95% of no-match S1 have a similar wrong candidate (≈7.5 each) — that is where singleton false positives come from.

| Signal | True (similar) | Copy of no business | Copy / other business's record |
| --- | ---: | ---: | ---: |
| Legal forms conflict | 0.2% (only PC/LP/LLP→LLC/INC) | 17.7% (LLC→INC/CORP/CO/LTD, PvtLtd→PublicLtd/LLP, INC→LLC/CORP…) | 20.8% |
| Legal form only on record | 7.6% | 18.3% | 20.9% |
| Legal form same | 47.3% | 25.9% | 14.5% |
| Exactly 1 extra word | 11% | 57% | 50% |
| House number equal | 72.5% | 7.3% | ~0 (99% no address) |
| House number shifted ≤10 | 1.9% | 43.7% | – |
| House number 2+ digits changed, same length | 2.4% | 18.8% | – |

- Copy words (always copies): southside, riverside, lakeside, initials (msh, rjm, jbv, hkg, rk, sg…), sweets, day.
  True-pair words: formerly, known, dba, aka, fka, nee, doing.
- Records with no address: similar candidates are true only 4.5% of the time (1 same-name rival) down to 0.4% (4+
  rivals); legal conflict → 0.2%. Mostly other businesses' address-less records → ambiguous by design.
- Invented one-word names never used in S1 names: true 54% when address ≥ 0.95 and same house number, 5% at ≥ 0.9.
- Whole train: 25% of unmatched records carry an S1's exact name; those copies differ by legal conflict (44%) or added
  legal form (34%), and by house number shifted ≤10 (46%) or 2+ digits (26%). Unmatched records do NOT come in groups
  (99.8% singletons), so no group-level logic is needed.

## Features v2 (trap features) — run p1-v3 stage 2
Added to stage 1 (both stages see them): legal category flags + per-side legal tags, house-number kind (equal,
|difference|, digit edits, prefix/suffix), differing words (counts, short word, alias marker), alias / frequency
(word counts, share of words unknown to S1 names, name/address frequency). Trained with XGBoost GPU.
Result: (to fill)
