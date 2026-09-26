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
| p1-v3-s2 | trap features (legal / house number / extra words / alias / frequency) + stage 2, XGBoost GPU | 0.9793 | 0.9794 | ? |
| p1-v3-s2ce | features v3 (French legal forms apart) + cross-encoder feature + 2× stage-2 data | ? | ? | ? |

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
Result (p1-v3-s2): stage 1 alone dev 0.9777 / fresh 0.9780; **stage 2 dev 0.9793 / fresh 0.9794**, precision
99.45%, recall 95.2%, no-match S1 given a link 1.2% (was 6.5%). Train and test pass ~20 min each on GPU.
Leaderboard: (to fill)

Rule from here: every model trains and predicts on the GPU.

### Where p1-v3-s2 still loses (dev panel 0.9793, threshold 0.741; work/plan1/p1-v3/error_analysis_s2/)
| Loss | Points |
| --- | ---: |
| True links search found but the model rejects (2,578 = 3.8% of found) | 1.30 |
| Wrong links on businesses with matches (344) | 0.36 |
| Search never found the record | 0.34 |
| No-match businesses given a link | 0.07 |

- Rejected true links: **59% have no address** (vs 2.3% of accepted true) — by trap analysis these are ambiguous by
  design (same-name address-less records of other businesses are true only ~4%); 27% had search rank > 10;
  8.5% only found by the fallback; invented names 15%; one extra word 15%; weak name 18%. Half are close calls
  (p 0.3–0.74), 23% confident rejects (aliases like "Dóvabrixorbi" with a matching address; "Paex Holdings"/"Paex LLC").
- Wrong links: half other businesses' records, half no-business copies; invented names 20%, other number
  differences 24%, weak names 23%, Indian-script 5%, websites 6%. Typical: "Great Diagnostics Inc" 301 vs 30 Poplar,
  "Novan Limited" vs "Novana Limited", "FK Des" vs "FM Des" (one-letter name changes = copies too).
- Simple "looks like an accepted record" rescue still hurts (adds ≥ as many wrong as right).

### Expected-F0.5 per-business decision (decide_ef.py): no gain
Calibrated (isotonic on C-select) + tuned alpha/gamma: dev 0.9791 vs 0.9793 global threshold, fresh 0.9800 vs
0.9794 — noise, and no-match S1 given a link doubles (1.2% → 2–3.6%). Keep the global threshold.

### France (15% of test S1, no training data): likely the biggest leaderboard lever
- Test p1-v3-s2 per country: France accepts 3.61 links/S1 and keeps 3.39 after one owner (6% of its links contested),
  while US/India accept 3.37 and keep 3.36–3.38. France's empty-answer rate is 5.0%, against 5.9% for US/India.
  So France over-links.
- If US/India score like the panel (0.979), France at 0.90 would make the leaderboard 0.967. Reaching 0.98 needs France ≈ 0.98.
- French names repeat a lot ("Tourcoing Sante" is the name of 14+ S1 in one city). There are about 15 cities, 4
  departments and 3 regions. S1 always writes the region; S2/S3 write it about 1/3 of the time and often write the
  department instead (Nord, Gironde, Loire-Atlantique, Pas-de-Calais).
- **Bug**: features v2 gave every French legal form (SARL/SAS/SASU/EURL/SA/SCI/SNC) ONE tag "FR", so planted swaps like
  SAS→EURL and EI→SCI looked "same". Copies seen in the test data: "XG Culturelle SAS 35 Av des Ajoncs" →
  "E.U.R.L. XG Culturelle 37 Av…"; "Jeux & Cie EI 108 Bd G. Pompidou" → "Jeux & Cie SCI 129 Bd…".
- **Features v3** (enrich.py):
  - Each French form gets its own tag, on French records only.
  - French-only dotted forms S.A./E.I. are handled.
  - Feature addresses drop French regions/departments and number markers (no 12 → 12; French n° → n 12 → 12).
- **Probe**: upload main + `nofrance` (France S1 empty). Then France score ≈ (main − nofrance)/0.1498 + ~0.056.
- Diagnostic: `plan1/france_check.py` measures legal-form change tables on near-identical pairs and accepted France
  links that have a form change.

### Leaderboard shape (2026-09-26)
- Clusters: 0.957–0.958 and 0.965 ± 0.001. Then an even spread from 0.970 to 0.978, a jump to 0.98, and the top 3 all at 0.988.
- Reading: tight clusters mean shared methods hitting the same traps, so gains come in steps (one trap type cracked
  at a time).
- The top 3 at 0.988 is probably the ceiling. Part of the data is ambiguous by design: address-less records whose
  name several businesses share (train has ~10 different businesses named "Obsidian, LLC").
- Realistic target: 0.985+.

### Raw-text features (features v3)
- raw_name_exact, raw_addr_exact, raw_name_ratio, raw_addr_ratio: exact text, case kept.
- Train examples: copies get re-formatted ("85 Wayne Avenue, …, NY" → "96 Wayne Avenue, …, New York"), while some
  true records keep the S1's exact address (the "Korbrixx D.B.A. Obsidian, LLC" alias).
- True records in one source share that source's typos (two S2 records both have "DEER PARK CIYT").
- True-record suffix words ("Center", "Service") differ from copy words ("Downtown", "Metro"…). The cross-encoder
  sees which word it is.

### Run p1-v3-s2ce (plan1/stage2ce.py + plan1/cross_encoder.py)
- Features v3, then stage 1 on the same 300k Fit sample, with out-of-fold scores.
- Stage 2 only looks at p1 ≥ 1e-3 (top 30 per S1). It trains on Fit (out-of-fold p1) plus gbm_extra, which is 300k new
  Fit-role S1 scored by the full stage-1 model, exactly as on test.
- New feature: a GPU cross-encoder (multilingual MiniLM-L12, Apache 2.0). It is fine-tuned on the ce_train pairs
  (300k other Fit-role S1) and adds its logit plus rank/max/gap/second/count context. A no-cross-encoder model is
  trained on the same rows for comparison.
- The test pass writes both outputs: `-s2ce` and `-s2ce-noce`.

### Locked plan (2026-09-26)
1. retrieve_extra.py: candidates for 600k more Fit-role S1 (CPU, background): 300k "ce_train", 300k "gbm_extra".
2. GPU text cross-encoder (multilingual MiniLM, Apache 2.0) trained on ce_train pairs; score = stage-2 feature.
3. Retrain stage 2 (XGBoost GPU) with cross-encoder score + gbm_extra data → panels → test → leaderboard.

## Next (ranked; goal 0.98+ on the leaderboard, 0.99 by day 3)
1. **Per-business decision instead of one global threshold**: calibrate stage-2 scores (isotonic on C-select), then
   for each S1 pick the set of candidates that maximises *expected* F0.5 (incl. the "nothing" option). No retraining;
   minutes. Expected +0.1–0.3.
2. **Learned (out-of-fold) tables** for trap words / legal-form swaps / one-letter name edits. Expected +0.1–0.2.
3. **More training data**: all 1.32M Fit-role S1 instead of 300k (needs ~2 h CPU search, run in background).
4. **Fine-tuned text cross-encoder on GPU** (multilingual MiniLM / XLM-R, MIT/Apache) as a stage-2 feature, for
   one-letter edits, aliases, typos, Indian script. Biggest remaining lever, ~1 day.
5. Search: deeper lists or a letter-group name+address channel for the 1.1% never found (0.34 pts).
