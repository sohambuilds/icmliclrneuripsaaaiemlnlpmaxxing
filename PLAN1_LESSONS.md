# Plan 1: final results and lessons

**25 September 2026.** Code: `plan1/`. Runs: `p1-baseline-v1` (Plan 1 as specified) and `p1-v2` (Plan 1 + quick fixes).

## 1. Scores

| Run | What it was | Audit panel (20k training S1) | Leaderboard (test) |
| --- | --- | ---: | ---: |
| `p1-baseline-v1` | Small cleanup → word TF-IDF search (top 50 per source) → 12 features → one LightGBM → one threshold (0.614) | 0.9123 | **0.8844** |
| `p1-v2` | + Indian-script dictionary, state names in one form, number-marker fix, French filler words, 28 features, 300k training businesses, one owner per record (threshold 0.6815) | 0.9523 | **0.9388** |

- **The leaderboard comes in below the Audit panel: −2.8 points for the baseline, −1.35 for v2.** Test has more unmatched records (about 40% vs 26%) and includes France, which has no training data. Expect **leaderboard ≈ panel − 1.5 points** until we learn otherwise.
- The gap shrank in v2, probably because France over-matched less (see section 3).

Detailed numbers:

| | Baseline | v2 |
| --- | ---: | ---: |
| Audit F0.5, US / India | 0.934 / 0.880 | 0.956 / 0.947 |
| Precision / recall of links | 96.4% / 84.4% | 98.5% / 89.9% |
| No-match businesses given a wrong link | 15.0% | 5.4% |
| Search found (all / India / Indian-script names / missing address) | 97.5% / 95.3% / 85.2% / 87.6% | 98.5% / 97.7% / 98.0% / 85.8% |
| Best possible score with these candidates | 0.991 | 0.995 |
| LightGBM: rounds, log-loss on Tune | 1,433, 0.0158 | 2,000 (hit the cap), 0.0095 |
| Name + house-number rule on the same candidates | 0.592 | 0.630 |

## 2. What moved the score

v2 changed many things at once, so we don't know exactly how much each fix gave. The evidence per fix:

1. **Indian-script word dictionary** (learned from Fit-role training pairs only; 1,205 words; covers 86% of Indian-script words in test).
   - Search for Indian-script names: 85% → 98%.
   - India's Audit score: 0.880 → 0.947.
   - The single biggest fix.
2. **Features that compare a candidate with the business's other candidates** (rank across both sources, share of the best score, gap to the best, and so on).
   - "Rank across both sources" became the second most important feature.
   - Wrong links on no-match businesses fell from 15% to 5.4%.
3. **One owner per record** (in training, each S2/S3 record belongs to at most one business).
   - On test, the baseline accepted 97,638 records for 2+ businesses.
   - The rule dropped 162,166 links, 71% of them in France.
   - Its leaderboard effect alone was not measured.
4. **Cleanup fixes**: legal forms removed anywhere, dotted forms (`L.L.C.`), invisible joiners in Indian words, state names in one form, number markers (`N°49`, `No.5`), French filler words.
5. **More training data and features**: 300k businesses instead of 100k, and 28 features instead of 12. Log-loss improved from 0.0158 to 0.0095.

## 3. What we learned about the data and the test

- **Remaining loss is mostly in the matching decision, not the search.** Search finds 98.5% of true links, but the model accepts only 89.9%, at 98.5% precision. The best possible score is 0.995, against 0.952 now.
- **Records with an empty address are the weakest search area** (85.8% found).
  - They also crowd the candidate lists: about 20% of all candidates have no address. With no address words, all their search weight sits on the name.
- **The model leans on the search score and rank**: about 73% of its importance in v2. Improving the search text improves the model.
- **France can only be watched through proxies**: matches per business, empty-answer rate, and conflicts per country on the test run.
  - The baseline over-matched France: 3.91 matches per business before one-owner and 3.46 after, with 11.4% of its links contested.
  - v2: 3.60 before and 3.36 after, closer to the roughly 3.3–3.4 the record counts suggest.
- **Some records were accepted for 100–293 businesses.** These are generic French names or records with no address. One-owner keeps one claim each, but even that claim is probably wrong.
- **Our samples can't see problems that only show up across all businesses.** The Audit panel showed 9 records claimed twice; the full test showed 97,638. Rules that depend on businesses competing (one owner, conflicts) can only be judged on full runs.
- **No shortcuts or leaks:** IDs and row order carry no information.

## 4. Process lessons

1. **Print before/after examples and sanity counts at every step.** They caught three silent bugs:
   - A 32-bit overflow in the sampling made the Fit sample 22,143 instead of 100,000.
   - Multi-word replacements never matched, so `L.L.C.` stayed `l l c`. The same bug made the EDA notebook's state-name step look weaker than it is.
   - Invisible joiners split Indian-script words in two.
2. **The official validator is not the whole portal.** A file that passed the validator was rejected until its rows were written in `test_source1.tsv` order. Now:
   - Always write rows in the test file's order: plain UTF-8, no BOM, no quotes, `\n` line endings.
   - Never open a submission in Excel.
3. **Same pipeline for train and test, always.** No test-only patches: every change is measured on held-out training businesses first.
4. **Version and fingerprint everything that is cached.** The cleanup version is in the file names, the model file has a fingerprint, test batches record which model made them, and scripts refuse stale inputs.
5. **Keep the expensive parts.** The test search takes about 3.2–3.75 hours, and test batches now save the search score and rank. A new model on the same cleanup only needs features and scoring. A cleanup change still means a new search.
6. **The Audit panel is now development data** (seen by both runs). For the next frozen comparison, draw a fresh panel from the roughly 420k unused Audit-role businesses.
7. **Change one thing at a time, or run ablations.** The methodology write-up will need each fix's contribution. Retraining without a feature group on saved candidates takes about 15 minutes.

Measured run times (Xeon w7-3445):

| Step | Baseline | v2 |
| --- | ---: | ---: |
| Cleaning train + test (24M records) | 23 s | 96 s (plus 7 s for the dictionary) |
| Training search | 160k queries, 18 min | 360k queries, 44 min |
| Features | 12M pairs, 24 s | 32M pairs, 2 min |
| LightGBM | 146 s | 806 s |
| Test run (1.73M businesses) | 194 min | 225 min |

## 5. Rules to keep following

- **Models:** any model used on the data must be MIT or Apache 2.0 and at most 8B parameters.
  - LightGBM is MIT.
  - anyascii is a text library, not a model.
  - The dictionary is learned from the training data only.
  - No outside lookups, and no Claude or any other outside service converting or labelling records.
- **Country:** an open set of labels. Only "same country or not" is used. French rules apply to every country, with no `if country == ...` code.
- **Leakage:** dictionaries and models are learned from the Fit role only; Tune, C-select and Audit businesses never feed learning.
- **Candidate file:** `candidate_pairs.tsv` must be exactly what the model scored, and every match must be in it.
- **Final zip:**
  - `output/` with both files.
  - `code/business_entity_resolution/`, holding `src/`, a README with one command per step, and pinned requirements.
  - The filled-in `Documentation_template.md`.

## 6. Next moves, cheapest first

1. **A name-only fallback search for records with no address** (currently 85.8% found).
2. **Let LightGBM train longer.** It hit the 2,000-round cap while still improving.
3. **Dictionary quality:**
   - Slightly lower similarity cutoff (`फूड` → `foods` was missed).
   - Check suspicious mappings (a Tamil word for "technology" → `real`).
4. **A stricter rule for records claimed by many businesses** (for example, drop them when claimed 10+ times). This can only be judged on the leaderboard.
5. **Ablations of the v2 fixes** for the methodology document.
6. **A fresh Audit panel** before the next frozen comparison.
7. **Bigger steps, only after 1–6:**
   - Competition features across businesses: how strongly other businesses claim the same record. This needs search for all 2.2M training businesses.
   - Embeddings, only on the links still missed.

## 7. Where things are

| What | Where |
| --- | --- |
| Code, one script per step | `plan1/step1_prepare.py` … `step5_infer.py`, `step5b_one_owner.py`, `rewrite_outputs.py` |
| Run selection | `AMC_RUN_ID` (default `p1-v2`); work files in `work/plan1/<run>/`, submissions in `output/plan1/<run>/` |
| Shared split (all plans) | `work/shared/splits/manifest.parquet`, fixed salts in `plan1/config.py` |
| Frozen settings per run | `work/plan1/<run>/frozen_config.json`, `audit_report.json`, `model_meta.json` |
| Indian-script dictionary | `work/plan1/p1-v2/translit.json` |
| EDA | `EDA_FINDINGS_ALL.md`, `eda.ipynb`, `eda_results/` |
| Run order | step 1 → 2 → 3 → 4 → 5, then `utils/validate_submission.py --check-ids` |
