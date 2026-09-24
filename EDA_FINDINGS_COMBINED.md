# What we learned from the business-matching data

**Combined findings from Fable and Codex | 25 September 2026**

The strongest direction is to search using **both the business name and the address**, then decide which of the suggested records really belong to the same business. Most businesses have several correct matches. Names alone miss too many records, and a similar address can still belong to a wrong match.

Fable's analysis explains the kinds of changes in the data. Codex's independent analysis confirms the main patterns and tests how well several search methods find the correct records. The promising results below are search experiments, not a final competition score.

**Latest result:** searching all training S2/S3 records for 1,000 sampled businesses found **98.36% of their true links** using combined name-and-address search. It found **99.82% in the US and 95.93% in India**. Improving the Indian-script and missing-address cases is the clearest next step.

## 1. What the task involves

**S1** is the reference list of businesses. For each S1 business, we must find **all** of its matching records in **S2** and **S3**. Some businesses have no matches. Several records from the same source can belong to one S1 business.

| Dataset | S1 reference businesses | S2 records to search | S3 records to search |
| --- | ---: | ---: | ---: |
| Training | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

Training contains the US and India. Test also contains France: **259,452 businesses, about 15% of test S1**.

S1 has much cleaner names and more complete addresses. S2 and S3 contain most of the spelling changes, missing information and other noise. Even S1 can reorder address components, so the first word or first number does not always mean the same thing.

Fable found no shared IDs or exact `(name, address)` combinations between training and test. Some business names repeat across the two sets, but at different addresses. Its checks found no useful matching shortcut in ID numbers or row order.

## 2. Finding several matches matters

Both analyses found the same overall pattern across all **2,206,821 training businesses**:

| Correct matches for an S1 business | Share of businesses |
| --- | ---: |
| None | 5.58% |
| One | 5.40% |
| Two | 17.00% |
| Three or more | 72.01% |

The average is **3.46 matches**, with a maximum of **11**. About **51.20%** of S1 businesses have multiple S2 matches, and **55.47%** have multiple S3 matches.

**Practical meaning:** keeping only the best record from each source would discard many correct answers. Codex calculated that even perfect choices under that restriction would cap the training score at **0.85736 out of 1**.

The competition calculates a score for each S1 business and averages those scores. It penalizes adding a wrong business more strongly than missing a correct match. For a business with three correct matches:

| Our answer | Score for that business |
| --- | ---: |
| All three correct matches | 1.000 |
| Two correct matches, no wrong ones | 0.909 |
| All three correct matches, plus one wrong one | 0.789 |

A business with no true matches scores 1 when we return an empty list and 0 when we return any match. These cases matter, but predicting an empty list for everyone would score only **0.05585** on training.

## 3. The labels give us two useful rules

The independent audit checked all **7,638,365 labeled matches**. Every referenced record exists, every S1 has a label row, and there are no repeated IDs inside the answer lists.

Two findings help structure the solution:

- **One S2 or S3 record belongs to at most one S1 business in the training labels.** A business can own several records, but an individual record is never shared by two businesses. We can use this to resolve competing claims. The score improvement from doing so has not yet been measured.
- **Country agrees on every true training match.** Country equality is useful for narrowing the search. The implementation must accept unfamiliar country labels, including France; this observation does not justify limiting the system to US and India.

There are **2,681,854 S2/S3 training records with no S1 match**, or **25.99%** of all S2/S3 rows. We call these *distractors*: records the system may encounter but should not assign to an S1 business.

Test contains more S2/S3 rows per S1 business. **If the average number of true matches stays the same as training**, roughly **40%** of test S2/S3 records would be distractors. This is an estimate from record counts; test answers are unavailable.

**Practical meaning:** evaluate with extra unrelated records in the search pool as well as the ordinary training distribution. A method that works among fewer distractors may make more false matches on test.

## 4. Names change in several different ways

Fable inspected a sample of **300,000 true matches**. Common changes include:

| Change | How common or what it looks like |
| --- | --- |
| Capitalization, accents, punctuation or company endings | About 49% of true pairs become identical after these are cleaned up. |
| Words added or removed | About 15.5%; examples include `Services`, `Dr`, `M/s`, and `formerly`. |
| Small spelling mistakes | About 11%; includes letter swaps and confusion between `l/1`, `O/0` and `S/5`. |
| Names written in Indian scripts | About 7% of all true pairs. |
| No shared name words | About 7%; includes website-style names and unrelated-looking aliases. |

These are selected findings, not a complete breakdown to add together.

Codex independently inspected **69,721 true matches belonging to 20,000 S1 businesses**. Basic name cleaning made **25.82%** identical; removing company endings at the end of the name raised that to **44.16%**; also ignoring word order raised it to **46.31%**. The exact percentages depend on the cleaning rules and sample. Both analyses show that simple cleanup helps substantially but leaves many matches unresolved.

### Indian scripts need to be preserved

Fable found a small, fairly consistent vocabulary: **1,463 distinct Indian-script tokens** in its dictionary experiment, with about **99% consistency** in their English mappings. That dictionary covered **93.5% of the tokens** in the inspected test names, and **80.8% of those names completely**. This makes a dictionary learned from the supplied training matches worth testing. Coverage is not the same as matching accuracy.

Codex's separate experiment supports converting these names into Latin letters as an additional view. On **4,971 true pairs with Indian-script names**, average name similarity on a 0-to-100 scale was approximately:

| Treatment | Average similarity |
| --- | ---: |
| Original text compared directly | 13 |
| Cleanup that removes non-Latin characters | 4 |
| Conversion into Latin letters | 71 |

This measures name similarity, not the percentage of records correctly matched.

**Practical meaning:** preserve the original name and create an additional Latin-letter version. When splitting text into words, keep vowel signs and other attached marks; removing them breaks Indian-script words. A learned dictionary must be built only from the training portion of each evaluation split.

### Website names and aliases need different treatment

Fable found website-style names in **3.54% of true pairs**. In about **68%** of those cases, removing the website ending and joining the cleaned S1 name words produces the same text. For example, `Allied Program` can become `alliedprogram.com`.

Another **1.77%** of true pairs use an unrelated-looking single-word alias. Codex also found that about **2%** of its sampled true pairs combine a very weak name match with a strong address match.

**Practical meaning:** allow strong address evidence to recover records whose names tell us very little. Requiring shared name words would lose real matches.

## 5. Addresses are powerful evidence, with important exceptions

In Fable's comparison of **200,000 true pairs and 200,000 randomly selected wrong pairs from the same country**, address-word similarity was the strongest individual signal tested:

- About 90% of true pairs scored at least **75 out of 100**.
- About 99% of random wrong pairs scored no higher than **55**.

That is encouraging, but random wrong pairs are usually easy. The confusing cases share a business name, street, city or other details.

Names themselves are highly repetitive. Codex found that **38.34%** of S1 businesses share their lowercase name with another S1. Fable's more extensive cleaning, including removal of company endings, puts the same-country repeated-name share at **49.5%**. A name match therefore needs supporting evidence.

Address changes include abbreviations, uppercase text, reordered components, missing street details, city/locality substitutions, inserted `NULL` or `N/A`, and Indian-script state names. S3 sometimes reduces an address to little more than a city and state.

### Missing addresses cannot mean automatic rejection

The full independent label audit found **337,018 correct matched records with missing addresses: 4.41% of all true links**. These records need to remain eligible when the remaining evidence is strong enough.

### Numbers help, but disagreements also occur in real matches

Fable found that leading house numbers agree in roughly **83–86%** of true pairs when both addresses start with a number.

Codex checked the first number anywhere in each address. Treating `007738` and `7738` as equivalent increased agreement from **68.58% to 72.19% of all sampled true pairs**. Even after that cleanup, **14.98% of all sampled true pairs had numbers on both sides that disagreed**.

**Practical meaning:** a different house or unit number should lower confidence, but should not automatically reject a pair. Compare the street, other numbers, name and remaining address together. Component reordering also makes the first extracted number an imperfect house-number estimate.

Postal codes offer limited coverage. Fable found no six-digit codes in the inspected Indian S1 addresses and five-digit codes in only about **11%** of US S1 addresses. Codex found a number of at least five digits on both sides in only **5.55%** of sampled true pairs. A long number is not necessarily a postal code; it can be a house number.

### Some wrong pairs look exceptionally convincing

One example from the independent hard-case file was checked again against the original records and answer labels while preparing this report:

| Field | S1 reference | S3 candidate |
| --- | --- | --- |
| Name | Ivenent Metal | Ivvenent Metal |
| Address | 22821 Lynwood Street, Buckeye, AZ | 22821 Lynwood St, Buckeye, Arizona |
| ID | S1-95240037 | S3-440603460 |

The training labels give this S1 **no matches**, and this S3 record is not listed as a match anywhere in the training answers. Here, the house number agrees and the addresses describe the same location.

This example shows why the difficult wrong pairs must be included in evaluation. Number conflicts explain many mistakes, but they do not explain every mistake. High name and address similarity is strong evidence, not a guarantee under the supplied labels.

## 6. What the search experiments actually achieved

Before deciding whether records match, the system needs a manageable shortlist. A correct record that never reaches this shortlist cannot be recovered by the final matching model.

Here, **“true matches found” means the percentage of all known correct links that reached the shortlist**. It does not measure how many wrong records remain on the list, or the final competition score.

### Simple search rules missed too many matches across the full training pool

Codex searched all **10,320,219 training S2/S3 records** for **2,000 sampled S1 businesses**, which have **7,022 true links**. Each method kept up to 50 suggested records from S2 and 50 from S3. These experiments did not filter by country.

| Search method | True matches found |
| --- | ---: |
| Same cleaned name, including reordered words | 45.88% |
| Same first six characters of the compacted name | 70.26% |
| A shared name word plus an address-number key | 58.40% |
| Combine those methods and rank by name, address and number evidence | **82.54%** |

Keeping every result from the combined rules, before shortening the lists, reached only **85.35%**. Increasing the final allowance to 100 records per source reached **83.55%**. Much of the loss happens because the rules never find the right record in the first place.

The combined method found **91.03%** of true US links but only **69.17%** of true Indian links at 50 per source.

**Practical meaning:** these shortcuts can help, but they are insufficient as the main search method.

### Full-pool text search is stronger, but India still needs work

Codex tested text search that compares words and short letter sequences in the combined name and address. The technical name is **TF-IDF search**: distinctive text receives more weight than very common text.

The first experiment used **1,000 S1 businesses with 3,542 true links** against **223,658 S2/S3 records**, about 2.2% of the full training pool. A subsequent benchmark searched **all 10,320,219 records for those same 1,000 businesses**.

| Search method | True links found: smaller pool | True links found: full pool | Average total shortlist per S1 in full pool |
| --- | ---: | ---: | ---: |
| Letter sequences from name and address | 99.44% | 97.35% | 100 |
| Words from name and address | 99.27% | 97.46% | 100 |
| Combine the two searches | **99.75%** | **98.36%** | **160.4** |
| Combine them, then shorten the merged list to 50 per source | **99.55%** | **97.80%** | **100** |

For individual searches, the allowance was 50 records per source. For the larger combined list, **each search received that allowance**, and their results were merged and deduplicated. Its 160.4-record average must not be described as a 50-record total budget.

The full-pool combined search found **3,484 of 3,542 true links**, missing **58**. The US and India results differ substantially:

| Country | S1 businesses searched | True links | Found with the larger merged list | Found after limiting to 50 per source |
| --- | ---: | ---: | ---: | ---: |
| US | 621 | 2,215 | **99.82%** | 99.50% |
| India | 379 | 1,327 | **95.93%** | 94.95% |

The current plan asks for at least 98% of true links to be found overall and in each country at the chosen shortlist limit. **This benchmark does not meet that requirement.** The larger merged list clears 98% overall but falls short in India; the list capped at 50 per source also falls below 98% overall.

The full run used generic conversion into Latin letters and a small address-abbreviation list. It did not use a learned Indian-script dictionary, country filtering, company-ending removal or leading-zero cleanup in its search text. These remain possible improvements to test, rather than improvements already demonstrated here.

### What the full-pool search missed

These figures use the larger merged list and the same **3,542 true links**:

| Kind of true pair | True links examined | Missed | Found |
| --- | ---: | ---: | ---: |
| S2/S3 name written in an Indian script | 233 | **33** | **85.84%** |
| S2/S3 address missing | 172 | **13** | **92.44%** |
| Weak name similarity, strong address similarity | 88 | 1 | 98.86% |
| First address numbers disagree | 650 | 15 | 97.69% |

The groups overlap, so the missed counts should not be added together. The “weak name, strong address” group uses scores below 50 for the name and at least 80 for address words; these were analysis labels, not search filters.

**Practical meaning:** Indian-script names account for 33 of the 58 missed links, making them a clear priority. Missing addresses need a stronger fallback using the remaining information. The search already handles most of this sample's weak-name, strong-address cases.

### Why the earlier 99.75% result needs context

The smaller pool deliberately included the known correct records with a sampled background. The full-pool run indexed every record and used the answer labels only to evaluate the results. It also recalculated word and letter-sequence weights across the full pool, so both competition from other records and text weights changed.

These full-pool results cover **1,000 sampled businesses**, not all 2.2 million S1 businesses. Repeated resampling of those businesses gave an estimated range of **97.88–98.80%** for the larger list's overall result. That range does not account for France or a different test-data distribution.

In the smaller experiment, adding a separate name-only search to the two combined-text searches found **no extra true links at the tested allowance**, while increasing the average list from **151 to 227** records. Adding address-only search after that found **two extra true links**, with an average list of **271**. These additions have not yet been tested in the full-pool benchmark, where they could behave differently.

## 7. France needs separate attention

These observations come from **Fable's inspection of the unlabeled test files**. Codex's independent EDA used training data only, so it did not measure French matching accuracy.

French names contain accents and company endings such as `SARL`, `SAS`, `SASU` and `EURL`. Addresses use abbreviations such as `r` or `r.` for `rue`, `av` for `avenue`, and number markers such as `n°`. Some text loses its first character, producing forms such as `oulevard` or `ordeaux`.

Fable found only about 20 cities and three regions in the French data. Those repeated location words provide limited help in telling businesses apart; street details and numbers become especially valuable.

**Practical meaning:** preserve accented text, support these address forms, and tolerate small spelling damage. Strong US and India results do not prove that France will work equally well.

## 8. The solution also has to fit in memory

With 50 candidates from each source, all training S1 businesses would produce **220,682,100 candidate pairs**. Storing just 50 numeric comparison values per pair, using four bytes each, would require about **44.1 GB**, before IDs, text and other working memory.

This is a calculation, not a measured peak-memory result. It is already larger than the **24 GB machine recorded in the independent audit**.

The completed full-pool text-search benchmark processed the data in pieces. For its **1,000 S1 queries against all S2/S3 records**, it recorded:

| Resource | Measured result |
| --- | ---: |
| Index construction, search and initial result calculation | 26.7 minutes |
| Peak memory across the benchmark process and its workers | 2.10 GB |
| Reusable files stored on disk | 17.87 GB |

This shows that the full search pool can be handled with modest working memory for the sampled query workload. It does **not** establish runtime for every S1 business or memory use for the final matching model.

**Practical meaning:** keep processing records in batches, reuse cleaned text, control shortlist sizes, and measure throughput before expanding to every business.

## 9. What to do next

1. **Use shared text cleanup.** Preserve original text; add cleaned and Latin-letter versions; handle company endings, website names, abbreviations, missing fields and leading zeros.
2. **Improve the gaps found in the full-pool search.** Start with Indian-script names and missing addresses. Test a training-only script dictionary and targeted fallback searches, then remeasure accuracy at the same shortlist allowance and cost.
3. **Evaluate under realistic conditions.** Keep businesses used for evaluation separate from those used to learn matching rules or dictionaries. Include convincing wrong pairs and test a larger distractor population. Do not manually insert correct answers into the final evaluation shortlists.
4. **Train a model to judge the shortlisted pairs.** Let it weigh names, street details, numbers, missing information and competing candidates together. Permit multiple correct records from each source.
5. **Measure the value of resolving competing claims.** The training labels support one owner per S2/S3 record, but the actual improvement needs testing.
6. **Check scale before a complete run.** The full target pool has been searched for 1,000 businesses. Measure throughput on larger query batches before generating results for every S1 business. Test larger text models only against specific mistakes that simpler changes leave unresolved.

## 10. How to read the evidence

This document combines existing results; it does not report a new model-training run. Training counts and independent search results were checked against their saved tables. The example in Section 5 was also checked directly against the original source records and labels.

| Evidence | What was examined |
| --- | --- |
| Independent structure and label audit | All 12,527,040 training source rows and all 7,638,365 true links. |
| Independent name/address checks | 69,721 true links from 20,000 sampled development businesses. |
| Independent full-pool simple search | 2,000 businesses against all 10,320,219 S2/S3 training records. |
| Independent smaller-pool text search | 1,000 businesses against 223,658 records, with known correct records deliberately included. |
| Independent full-pool text search | The same 1,000 businesses against all 10,320,219 S2/S3 records; answer labels used only for evaluation. |
| Fable's training/test comparison and France observations | Existing data diagnostics in the main experiment log; test has no answer labels. |

The independent sampled experiments excluded a reserved group of roughly 20% of S1 businesses. Global structural counts did include that group. No final matching-model score or French accuracy result is established by these reports.

The full-pool text-search benchmark is recorded under `eda/full_sparse_20260925/`. Its saved verification report passes checks of the original query identities, source counts, candidate IDs, rankings and independently recalculated search results. Its latest completed results are included above.

### A few numbers need careful wording

- **Training distractors:** the short summary's “27%” is approximate. The exact combined share is **25.99%**; S2 is 26.64% and S3 is 25.37%.
- **Test distractors:** “about 40%” depends on assuming the same average number of matches as training. It is not a measured test-label statistic.
- **Repeated names and cleaned-name matches:** the two reports use different cleaning rules, grouping definitions and samples. Their percentages should not be treated as identical measurements.
- **Shared words are not proof of a match:** Fable found that about **99.99%** of true training pairs share at least one name word or address word. It also found fewer than 0.1% of sampled true pairs with both similarities below 50. These suggest that most correct pairs retain useful clues; they do **not prove an achievable score or recall above 99%**, because wrong pairs can share those clues too.
- **Search success is not matching accuracy:** **99.75%** came from the smaller pool; **98.36%** came from the full pool with a larger merged shortlist; **97.80%** came from the full pool capped at 50 per source. A final system must also reject wrong candidates and handle businesses with no matches.

## Source files and examples

- [Fable's detailed diagnostics and the shared experiment log](D:/PROJECTS/amazon_ml_challenge/EXPERIMENTS.md)
- [Codex's independent findings](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/FINDINGS.md) and [experiment scope](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/README.md)
- [Competition task and scoring rules](D:/PROJECTS/amazon_ml_challenge/student_resource/README.md)
- [Full label totals](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/label_summary.tsv), [label integrity checks](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/integrity.tsv), and [verification results](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/verification.txt)
- [Full-pool simple-search results](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/full_blocking_curves.tsv) and [smaller-pool text-search results](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/sparse_probe_curves.tsv)
- [Number comparisons](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/numeric_sensitivity_summary.tsv) and [missing-address counts](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/full_missingness_by_label.tsv)
- [Fable's 30 difficult true pairs and 30 near-misses](D:/PROJECTS/amazon_ml_challenge/eda/out/hard_samples.txt)
- [Codex's 30 difficult true pairs](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/hard_positives_raw.tsv) and [30 convincing wrong pairs](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/near_misses_raw.tsv)
- [Full-pool text-search benchmark description](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/README.md)
- [Completed full-pool text-search results](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/RESULTS.md), [detailed search results](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/recall_curves.tsv), and [verification](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/verification.txt)
- [Breakdown of full-pool search mistakes](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/retrieval_error_slices.tsv) and [the 58 missed true pairs](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/missed_true_pairs_raw_k50.tsv)
