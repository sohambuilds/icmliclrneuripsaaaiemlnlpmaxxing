# What we learned from the business-matching data

**Combined findings from Fable, Codex and the `eda.ipynb` notebook | 25 September 2026**

## Words used in this document

| Word | Meaning |
| --- | --- |
| S1, S2, S3 | The three sources. S1 is the clean reference list; for each S1 business we must find its records in S2 and S3. |
| True link / true pair | An S1 business and one of its correct S2/S3 records, according to the training answers. |
| Distractor | An S2/S3 record that belongs to no S1 business. |
| Score | The competition score (F0.5): worked out for each S1 business, then averaged. It counts precision (not adding wrong records) twice as much as recall (finding all correct records). |
| Shortlist / candidate search | The first step: for each S1 business, pull a manageable list of possible S2/S3 records. A record missing from the shortlist can never be matched later. |
| Found (recall) | The share of true links that made it onto the shortlist or into the answer. |
| Similarity score (0–100) | How alike two texts are. *Whole-text* compares the strings as they are; *word-order-free* sorts the words first; *word-overlap* checks how much of the shorter text's words appear in the longer one. |
| Weighted text search (TF-IDF) | A search that scores records by shared words or short letter groups, giving rare ones more weight than common ones. |
| Gradient boosting | A standard machine-learning model that learns how to combine many similarity scores into one yes/no probability. |
| Separation score (AUC) | How well one score tells true matches apart from wrong candidates: 0.5 is no better than chance, 1.0 is perfect. |
| Transliteration | Writing the same word in another alphabet by its sound (प्राइवेट → "private"). Different from translation, which changes the word. |

Sources of each finding: **Fable** and **Codex** are the two earlier analyses; **Notebook** is `eda.ipynb` (tables and charts in `eda_results/eda_outputs/`).

## The short version

1. **Search with name and address together, then decide.** Most businesses have several correct records, names alone repeat a lot, and addresses alone can mislead.
2. **The best search so far finds 98.36% of true links** (US 99.82%, India 95.93%). India needs work: Indian-script names and missing addresses cause most misses. *(Codex)*
3. **Fixed rules top out around 0.6.** The best rule, same cleaned name and same house number, scores 0.599 with 95% precision. *(Notebook)*
4. **A learned model beats hand-written rules clearly.** On the same shortlists, gradient boosting reached 94% of the best possible score; hand rules reached about 75%. *(Notebook)*
5. **Indian-script names are English words written by sound**, not translations. Generic letter conversion is too rough (1.6% become identical). A word dictionary learned from the training pairs should come first. Claude or any other outside model should not convert the records. *(All)*
6. **Clean names with a broad list of company endings.** Removing them anywhere in the name is the single biggest cleaning step (+25 points of identical names). *(Notebook)*
7. **Put state names into one form.** S1/S2 and S3 write US and Indian states in opposite ways. *(Notebook)*
8. **Compare addresses by overlap, not equality.** Addresses mostly differ because parts are missing. *(Notebook)*
9. **Be cautious.** One wrong extra record on every business costs far more than one missed record. Test also has more distractors (about 40%) than training (26%). *(All)*
10. **Duplicates inside S2/S3 can help.** For 76% of hard-to-match records, an easier record of the same business is very similar to it. *(Notebook)*

## 1. The data

**S1** is the reference list of businesses. For each S1 business, we must find **all** of its matching records in **S2** and **S3**. Some businesses have no matches. Several records from the same source can belong to one S1 business.

| Dataset | S1 reference businesses | S2 records to search | S3 records to search |
| --- | ---: | ---: | ---: |
| Training | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

- Training contains the US and India. Test also contains France: **259,452 businesses, about 15% of test S1**.
- S1 has much cleaner names and more complete addresses. S2 and S3 hold most of the spelling changes, missing information and other noise.
- Even S1 can reorder address parts, so "the first word" or "the first number" does not always mean the same thing.
- Empty addresses: none in S1; about **3.3%** of S2/S3 in training and **2.7%** in test. *(Notebook)*
- No ID is shared between training and test, and no exact (name, address) combination repeats across them. Some business names repeat across the two sets, but at different addresses. *(Fable)*
- **No shortcut in IDs or row order.** Matched and unmatched records have the same ID-number and file-position patterns, and true pairs are no closer in ID or row position than random pairs. *(Fable; confirmed by Notebook)*

## 2. How the score works

Most businesses have several correct records:

| Correct matches for an S1 business | Share of businesses |
| --- | ---: |
| None | 5.58% |
| One | 5.40% |
| Two | 17.00% |
| Three or more | 72.01% |

- The average is **3.46 matches**, with a maximum of **11**. US and India look the same.
- **51.20%** of businesses have two or more S2 matches; **55.47%** have two or more S3 matches.
- Keeping only the best record from each source caps the training score at **0.857**, even with perfect choices. *(Codex; Notebook agrees)*

The score punishes a wrong record more than a missed one. For one business with three correct records:

| Our answer | Score for that business |
| --- | ---: |
| All three correct records | 1.000 |
| Two correct records, no wrong ones | 0.909 |
| All three correct records, plus one wrong one | 0.789 |

A business with no correct records scores 1 for an empty answer and 0 for any answer. Answering "nothing" for every business scores only **0.056**.

The same rules applied to the whole training set *(Notebook)*:

| Scenario (applied to every training business) | Training score |
| --- | ---: |
| Answer nothing | 0.056 |
| Exactly one correct record per business | 0.696 |
| One correct record per source (best S2 + best S3) | 0.857 |
| All correct, plus one wrong record on **every** business | 0.752 |
| All correct, except one missed on each business with several | 0.925 |
| All correct, but one wrong record on each no-match business | 0.944 |

**What it means:** one extra wrong record everywhere (0.75) hurts far more than one missed record everywhere (0.93). When unsure, leave the record out. Getting every no-match business wrong costs at most about 0.056.

## 3. What the answer labels tell us

All **7,638,365 true links** were checked: every listed record exists, every S1 business has a row, and no answer list repeats an ID. *(Codex)*

- **Each S2/S3 record belongs to at most one S1 business.** A business can own several records, but a record is never shared. We can use this to settle two businesses claiming the same record. The gain has not been measured yet.
- **Country is the same on every true link.** Only compare records with the same country label. The code must accept any country label (France included) and must not be limited to US and India.
- **2,681,854 training S2/S3 records (25.99%) are distractors**: S2 26.64%, S3 25.37%.
- **Test probably has more distractors.** Test has more S2/S3 records per S1 business. If businesses have as many matches as in training, about **40%** of test S2/S3 records are distractors. This is an estimate from counts, not a measured fact. By country *(Notebook)*:

| Test country | Estimated distractors in S2 | Estimated distractors in S3 |
| --- | ---: | ---: |
| US | 40.8% | 39.1% |
| India | 41.3% | 39.8% |
| France (using the US/India match rate) | 38.2% | 36.6% |

**What it means:** test our methods with extra unrelated records added to the pool, not only on the normal training mix. A method that looks precise among fewer distractors can make more wrong matches on test.

## 4. Names

### How names change between S1 and S2/S3

From a sample of **300,000 true links** *(Fable)*:

| Change | How common or what it looks like |
| --- | --- |
| Capital letters, accents, punctuation or company endings | About 49% of true pairs become identical once these are cleaned. |
| Words added or removed | About 15.5%; for example `Services`, `Dr`, `M/s`, `formerly`. |
| Small spelling mistakes | About 11%; letter swaps and `l/1`, `O/0`, `S/5` confusion. |
| Names written in Indian scripts | About 7% of all true pairs. |
| No shared name words | About 7%; website-style names and unrelated-looking aliases. |

These overlap and should not be added up.

From **69,721 true links of 20,000 businesses** *(Codex)*: basic cleaning made **25.82%** identical; also removing company endings at the end of the name made **44.16%**; also ignoring word order made **46.31%**.

### Each cleaning step, measured on all true links *(Notebook)*

Steps are added one after another; the table shows the share of all **7,638,365** true links whose names become identical.

| Step | All | US | India | Indian-script S2/S3 name |
| --- | ---: | ---: | ---: | ---: |
| Raw text | 4.6% | 5.9% | 2.7% | 0% |
| Lowercase | 15.8% | 19.0% | 10.9% | 0% |
| Remove punctuation | 22.7% | 27.2% | 16.1% | 0% |
| Remove junk (`www…`, `(ID: …)`, `m/s`, `dba`, phone numbers) | 23.1% | 27.5% | 16.6% | 0% |
| Remove accents | 27.2% | 32.2% | 19.6% | 0% |
| Remove company endings and `and/the/of`, **anywhere in the name** | **52.3%** | 56.6% | 45.9% | 0% |
| Ignore word order and repeated words | 55.7% | 61.0% | 47.7% | 0% |
| Convert to English letters with a generic tool (anyascii) | 55.8% | 61.0% | 48.0% | 1.6% |
| Word-order-free similarity ≥ 90 | 64.0% | 69.8% | 55.3% | 6.2% |
| Word-overlap similarity ≥ 90 | **75.0%** | 81.3% | 65.7% | 6.4% |

- Removing company endings is by far the biggest step (+25 points). Junk removal adds little; accents add 4 points.
- After everything, **27% of Indian S2 pairs and 22% of Indian S3 pairs** still score below 80, against about **11%** in the US.
- The three analyses use different rules and samples (46.31%, about 49%, 55.8%), so their numbers are not the same measurement.

### Noise in names, by source *(Notebook)*

Share of names showing each pattern (training; test is almost identical):

| Pattern | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| Company ending moved to the front (`LLC Moncada…`) | ~0% | 3.0–4.1% | 2.9–4.6% |
| Website used as the name | 0% | 2.7–4.0% | 2.9–3.8% |
| Trade-name markers `dba` / `aka` / `fka` / `t/a` | 0% | **0%** | 1.4–2.1% |
| Fake accents (`Léarning`) | 0% | 4.4–6.7% | 5.3–6.8% |
| Digit inside a word (`1ife`, `Gl0bal`) | 0% | 1.3–1.9% | 1.4–1.9% |
| Junk at the start (`--`, `<<`, `#`) | ~0.1% | 0.9–1.4% | 1.0–1.4% |
| `(ID: 12345)` or a phone number | 0% | 0.5–0.7% | 0.6% |
| Repeated word (`King King`) | 0.4–2.5% | 3.1–5.6% | 3.3–5.4% |
| English and Indian script mixed in one name (India) | 0% | 0.9% | 1.7% |

Trade names appear **only in S3**. Junk also includes wrappers such as `| www.x.com`, `>>` and bracketed words like `[Corp]` or `(Limited)`.

### Words that most often appear on only one side of a true pair *(Notebook)*

- Company endings: `limited`, `ltd`, `private`, `pvt`, `llc`, `inc`, `corp`, `corporation`, `co`, `company`, `llp`, `lp`, `pc`, `pllc`, `incorporated`.
- The same endings in Indian scripts: `लिमिटेड`, `प्राइवेट`, `प्रा`, `लि`, `एलएलपी`, and their Tamil, Telugu, Kannada, Bengali, Gujarati and Malayalam forms.
- Titles: `sri`, `shri`, `smt`, `mr`, `dr`.
- Trade-name words: `formerly`, `doing business as`, `dba`.
- Generic words: `center`, `services`, `service`, `group`, `care`, `associates`, `partners`, `india`, `trading`.

### Website names and aliases

- **Website names:** 3.54% of true pairs. In about **68%** of them, removing the ending and joining the cleaned S1 name words gives the same text (`Allied Program` → `alliedprogram.com`). *(Fable)*
- **Unrelated aliases:** another **1.77%** of true pairs use an unrelated-looking one-word name. *(Fable)*
- About **2%** of sampled true pairs have a very weak name match but a strong address match. *(Codex)*

**What it means:** strong address evidence must be able to match a record even when the name says little. Requiring shared name words would lose real matches.

### Many businesses share a name

- **38.34%** of S1 businesses share their lowercase name with another S1 business. *(Codex)*
- With endings removed and within the same country: **49.5%** *(Fable)*; **48.8% in the US and 54.8% in India** *(Notebook)*.
- **19.6% of US and 33.5% of Indian businesses** are in groups of ten or more with the same cleaned name. The top US name, `meridian`, is shared by **576** businesses. *(Notebook)*
- Adding the house number almost removes this in the US (**0.9%** still shared) but not in India (**10.0%**). *(Notebook)*

**What it means:** a name match alone is weak evidence. The hardest wrong candidates are other businesses with the same name.

## 5. Indian-script names

### What they are

- S1 names are **always in English letters**. Indian-script names appear only in Indian S2/S3 records: about **23%** of Indian S2 names and **13%** of Indian S3 names contain some. Overall that is about **7% of all true links**. *(Notebook, Fable)*
- Nine scripts appear. Hindi/Marathi script (Devanagari) is the largest, in about 13% of Indian S2 names. Bengali, Gujarati, Punjabi, Odia, Tamil, Telugu, Kannada and Malayalam each appear in about 1–2%. *(Notebook)*
- **They are transliterations, not translations.** English business words are written by their sound: `प्राइवेट लिमिटेड` = "private limited", `मार्केटिंग` = "marketing", `इंजीनियरिंग` = "engineering", `ट्रेडिंग` = "trading", `एक्सपोर्ट्स` = "exports", `फूड्स` = "foods". None of the frequent words were Hindi translations (such as "vyapar" for trading). Only the most frequent words were checked, so a few translated names may exist. *(Notebook)*
- About 1–2% of Indian S2/S3 names mix scripts (`Sun पावर Provision`), so conversion must work **word by word**. *(Notebook)*

### How well conversion works today

On **4,971 true pairs with Indian-script names**, average name similarity (0–100) *(Codex)*:

| Treatment | Average similarity |
| --- | ---: |
| Compare the original text directly | 13 |
| Clean up by deleting non-English letters | 4 |
| Convert into English letters | 71 |

Converted names get close, but rarely close enough to decide *(Notebook)*:

- After generic conversion (anyascii) plus all other cleaning, only **1.6%** of these pairs become identical, and **6.4%** reach a word-overlap similarity of 90.
- The generic tool has consistent habits:
  - Company endings come out as `praivet`, `praibhet`, `piraivet`, `praivrr` (private); `limitet`, `limirrd`, `limtid` (limited); `pra` + `li` (प्रा. लि.); and `elelpi` (LLP).
  - It drops vowels: `globl`, `prodkts`, `sistms`, `hotl`.
  - It writes "ng" as "mg": `marketimg`, `tredimg`, `imjiniyrimg` (engineering).
- Splitting words on "anything that is not a letter or digit" breaks Indian-script words, because their vowel signs count as separate marks. The word pattern must keep them (`[\p{L}\p{M}\p{N}]+`). *(Fable, Notebook)*

### A word dictionary learned from the training pairs

- The vocabulary is small and repeats. A dictionary experiment found **1,463 distinct Indian-script words**, whose English equivalents were **about 99% consistent**. *(Fable)*
- That dictionary covered **93.5% of the words** in the Indian-script test names, and **80.8% of those names completely**. Coverage is not the same as matching accuracy. *(Fable)*
- Its target is **S1's own spelling**. That matters more than a "correct" spelling. For example, it learns `Lakshmi` rather than `Laxmi` and `Sri` rather than `Shree`, whichever S1 uses.

### Should an LLM (such as Claude) convert all the Indian-script text?

**No, not on the records.**

- **Rules:** the final model must be MIT or Apache 2.0 licensed and at most 8 billion parameters. Claude is neither. If it converts test names, it becomes part of the solution. Sending test data to an outside paid service is also the kind of thing reviewers check.
- **Reproducibility:** the final package must regenerate the output from its own code. A hand-driven chat session cannot be reproduced.
- **Consistency:** an LLM writes its own preferred spelling, which may not match S1's. The learned dictionary gives S1's spelling directly.
- **Size:** training alone has about **280,000 distinct Indian-script names**, and test about as many. That is far too many for direct prompting.

Claude can still help **during development, on training data only**: review a sample of dictionary entries and the pairs that still fail, and suggest fix rules. Those rules then become ordinary code.

### Recommended approach, in order

1. **Learned dictionary (main step).** From training pairs whose S2/S3 name contains Indian script, line up the words with the S1 name. Count which English word each Indian-script word becomes, and keep mappings seen at least 3 times with at least 80% agreement. The output is S1's spelling. Build the dictionary only from the training part of each evaluation split, and save it with the model.
2. **Fallback for unknown words.** Use generic conversion (anyascii) plus fixes: "mg" → "ng" where it stands for "ng", and the ending spellings above (praivet → private, limitet → limited, elelpi → llp).
3. **Optional allowed model, only if 1–2 leave a big gap.** Run it as a script inside the pipeline on unknown words only, and cache the results. Candidates:
   - AI4Bharat IndicXlit, built for Indian-script ↔ English transliteration.
   - Qwen2.5-7B-Instruct (Apache 2.0, 7.6B parameters).
   
   Confirm the licence on the model card and record it. Keep the model only if it beats step 2 on held-out pairs.
4. **Keep both versions** of every name (original and converted) and compare with similarity scores, not equality, since spellings still vary.
5. **Addresses:** the Indian-script text there is mostly **16 state names**, so a small hand-written list covers it. Other address words go through the same dictionary and fallback.

**How to check it worked:** on held-out training pairs with Indian-script names, measure two things:
- **Word-overlap similarity of 90 or more:** today 6.4% of these pairs reach it.
- **Search recall for those links:** today 85.84% in the full-pool search.

## 6. Addresses

### Addresses are strong evidence, with exceptions

- Comparing **200,000 true pairs** with **200,000 random wrong pairs** from the same country, address word similarity was the strongest single signal *(Fable)*:
  - About 90% of true pairs scored at least **75**.
  - About 99% of random wrong pairs scored **55 or less**.
- Random wrong pairs are easy, though. The hard wrong candidates share a name, street or city.
- In India the address is more reliable than the name: **86%** of true pairs score at least 90 on address words, against **66%** on name words. In the US both are about **81%**. *(Notebook)*
- At least one of name or address reaches 90 in **96%** of true pairs. Both are below 70 in only **0.45%** (India) and **0.21%** (US). *(Notebook)*

### How each source writes addresses *(Notebook)*

| Source | US state | Indian state | Capital letters |
| --- | --- | --- | --- |
| S1 | Code (`TX`) | Full name (`Maharashtra`) | Normal |
| S2 | Code (`TX`) | Full name, or Indian script (`महाराष्ट्र`) | **~90% of US addresses ALL CAPS**; 24% in India |
| S3 | Full name (`Texas`) | Code (`MH`), or Indian script | Normal |

Other address changes:
- **Abbreviations and missing parts:**
  - Abbreviations vary: `ST`/`Street`, `RD`/`Road`.
  - S2 sometimes wrongly expands `ST` to `SAINT` (`MAIN SAINT`).
  - Street details and PIN codes go missing.
  - S3 sometimes reduces an address to little more than city and state.
- **Order, place names and numbers:**
  - Parts appear in a different order.
  - A locality or district replaces the city.
  - Numbers carry leading zeros (`002078`) or extra marks (`##68`, `135.`).
- **Other text:**
  - Landmarks appear (`Near SBI ATM`, `Opp.`, `C/O`).
  - Typos (`INDIANAPOLS`, `BUFFFALO`).
  - `NULL` or `N/A` inserted.
  - Indian-script state names: 16 states; about 23% of Indian S2/S3 addresses contain some Indian script.

*(Fable, Codex, Notebook)*

### Each cleaning step, measured on all true links *(Notebook)*

| Step (added on top of the previous ones) | All | US | India |
| --- | ---: | ---: | ---: |
| Raw text | 2.2% | 2.3% | 2.1% |
| Lowercase | 7.4% | 7.8% | 6.8% |
| Remove punctuation and leading zeros | 8.6% | 9.2% | 7.7% |
| Standard abbreviations (`street`/`saint` → `st`, `avenue` → `ave`…) | 12.5% | 15.7% | 7.8% |
| State names in one form | 21.3% | 27.5% | 12.0% |
| Convert Indian-script text (state names first) | 22.8% | 27.5% | 15.7% |
| Ignore the order of parts | 34.8% | 39.6% | 27.6% |
| All words of one address appear in the other | 60.8% | 60.5% | 61.2% |
| Word-overlap similarity ≥ 90 | **83.1%** | 81.0% | 86.3% |

- Putting state names into one form alone lifts S3's identical addresses from **4.3% to 21.2%**.
- The big later gains come from ignoring order and allowing one address to be a shortened version of the other. Addresses mostly differ by **missing parts**, not different content.

### Missing addresses are not a reason to reject

**337,018 true links (4.41%)** have a missing address on the S2/S3 side. These records must stay eligible when the name and other evidence are strong. *(Codex)*

### House numbers help, but real matches disagree too

- When both addresses start with a number, the numbers agree in about **83–86%** of true pairs. *(Fable)*
- Using the first number anywhere: agreement rises from **68.58% to 72.19%** of true pairs when `007738` and `7738` count as equal. Even then, **14.98%** of true pairs have numbers on both sides that disagree. *(Codex)*
- By country, first number after cleaning *(Notebook)*:

| Country | Same first number | Different first number | Missing on one side |
| --- | ---: | ---: | ---: |
| US | 76–77% | 10–11% | 12–14% |
| India | 60–61% | 18–19% | 20–22% |

House numbers disagree about **twice as often in India**. Because parts get reordered, "the first number" is only an estimate of the house number.

**What it means:** a different house or unit number should lower confidence, but not reject a pair on its own. Weigh the street, other numbers, name and the rest of the address together.

### Postal codes are rare

- Indian S1 addresses: no six-digit codes found. *(Fable)*
- US S1 addresses: five-digit codes in about **11%**. *(Fable; Notebook ~10–11%)*
- France: five-digit codes in about **0.5%**. *(Notebook)*
- A number of five or more digits appears on both sides in only **5.55%** of true pairs, and it is not always a postal code: it can be a house number. *(Codex)*

### Some wrong pairs look completely convincing

This pair was checked against the original records and labels *(Codex)*:

| Field | S1 reference | S3 candidate |
| --- | --- | --- |
| Name | Ivenent Metal | Ivvenent Metal |
| Address | 22821 Lynwood Street, Buckeye, AZ | 22821 Lynwood St, Buckeye, Arizona |
| ID | S1-95240037 | S3-440603460 |

The labels give this S1 business **no matches**, and this S3 record matches nobody, even though the name, house number and address all agree.

**What it means:** include hard wrong pairs in testing. High name and address similarity is strong evidence, but under these labels it is not a guarantee.

## 7. How far fixed rules go *(Notebook)*

Each rule links an S1 business to every same-country S2/S3 record with an identical cleaned key. Scored with the real competition score on **all 2,206,821 training businesses** (keys shared by more than 50 S2/S3 records were skipped):

| Rule | Score | Precision | Recall | No-match businesses right |
| --- | ---: | ---: | ---: | ---: |
| Answer nothing | 0.056 | – | 0% | 100% |
| Same cleaned name | 0.447 | 62.2% | 45.7% | 51.4% |
| Same cleaned address | 0.518 | 92.6% | 34.6% | 93.1% |
| Same cleaned name + same full cleaned address | 0.357 | **99.98%** | 18.4% | 99.97% |
| **Same cleaned name + same house number** | **0.599** | 94.9% | 39.2% | 94.2% |
| Same, but only when one S1 business has that key | 0.591 | 98.0% | 38.1% | 97.2% |

**What it means:** fixed rules top out around **0.6**. They are precise but find well under half of the true links. Name + house number makes a good high-precision signal for the model, not a full solution.

## 8. Finding candidates (the shortlist)

"Found" here means the share of true links that reached the shortlist. It says nothing about how many wrong records are on the list, or about the final score.

### Simple keys miss too much or cost too much

Codex searched all **10,320,219 training S2/S3 records** for **2,000 businesses** (**7,022 true links**). Each method kept up to 50 records from S2 and 50 from S3, without filtering by country:

| Search method | True links found |
| --- | ---: |
| Same cleaned name, word order ignored | 45.88% |
| Same first six letters of the name with spaces removed | 70.26% |
| A shared name word plus an address-number key | 58.40% |
| Combine those and rank by name, address and number evidence | **82.54%** |

- Keeping every result before shortening reached only **85.35%**; allowing 100 records per source reached **83.55%**. Most losses happen because the rules never find the right record at all.
- The combined method found **91.03%** of US links but only **69.17%** of Indian links. *(Codex)*

The notebook measured plain word keys on **all 7,638,365 true links** (same country only). Cost is the number of comparisons a key creates per S1 business. A word's *frequency* is how many same-country S2/S3 records contain it; using only rare words keeps the cost down.

| Key | True links found | Comparisons per business |
| --- | ---: | ---: |
| Same cleaned name | 55.8% | 51 |
| Same house number + first 3 letters of the cleaned name | 54.4% | 88 |
| A shared name word (frequency ≤ 1,000) | 44.3% | 136 |
| A shared address word (frequency ≤ 1,000) | 70.2% | 458 |
| **A shared name or address word (frequency ≤ 1,000)** | **83.2%** | **594** |
| A shared name or address word (frequency ≤ 10,000) | 97.8% | 10,866 |
| A shared name or address word (frequency ≤ 100,000) | 99.86% | 106,530 |
| Any shared name or address word | 99.99% | 1,652,741 |

- Address words beat name words at the same cost (**92% vs 69%** at frequency ≤ 10,000).
- Almost every true pair (99.99%) shares at least one name or address word. Using that directly costs far too much. *(Fable, Notebook)*

**What it means:** simple keys can support the search but cannot be the main method. Plain word keys need about 11,000 comparisons per business to reach 98%. Weighted text search reaches it with about 160.

### Weighted text search works much better, but India still needs work

Codex tested weighted text search (TF-IDF) on the combined name and address, using both whole words and short letter groups. First on **1,000 businesses (3,542 true links)** against a smaller pool of **223,658 records**, then against **all 10,320,219 records**:

| Search method | Found: smaller pool | Found: full pool | Average shortlist per business (full pool) |
| --- | ---: | ---: | ---: |
| Letter groups from name and address | 99.44% | 97.35% | 100 |
| Words from name and address | 99.27% | 97.46% | 100 |
| Both searches combined | **99.75%** | **98.36%** | **160.4** |
| Both combined, then cut to 50 per source | **99.55%** | **97.80%** | **100** |

Each single search kept 50 records per source. The combined list merges both searches' lists, so its 160.4 average is **not** a 50-per-source budget.

The full-pool combined search found **3,484 of 3,542** true links, missing **58**:

| Country | Businesses searched | True links | Found (combined list) | Found (cut to 50 per source) |
| --- | ---: | ---: | ---: | ---: |
| US | 621 | 2,215 | **99.82%** | 99.50% |
| India | 379 | 1,327 | **95.93%** | 94.95% |

The goal is at least 98% overall **and in each country**, and this search does not meet it yet. The combined list passes overall but not in India, and the list cut to 50 per source is below 98% overall.

The run did **not** yet use several things, which remain untested improvements:
- A learned Indian-script dictionary.
- Filtering by country.
- Company-ending removal.
- Leading-zero cleanup.

It used only generic conversion and a small abbreviation list.

What it missed (the groups overlap, so the missed counts don't add up):

| Kind of true link | Examined | Missed | Found |
| --- | ---: | ---: | ---: |
| S2/S3 name in an Indian script | 233 | **33** | **85.84%** |
| S2/S3 address missing | 172 | **13** | **92.44%** |
| Weak name (< 50), strong address (≥ 80) | 88 | 1 | 98.86% |
| First address numbers disagree | 650 | 15 | 97.69% |

**What it means:** Indian-script names cause 33 of the 58 misses, so they come first. Missing addresses need a fallback that searches on the name. Weak-name, strong-address cases are already handled well.

Context for these numbers *(Codex)*:
- **The smaller pool is easier.** It deliberately contained the known correct records. The full-pool run indexed every record and used labels only to score, and its word weights were recalculated over the full pool.
- **Sampling uncertainty.** The results cover 1,000 businesses, not all 2.2 million. Resampling puts the combined-list result between **97.88% and 98.80%**. That range does not cover France or differences in test data.
- **Extra searches barely helped.** In the smaller pool, adding a name-only search found **no extra links** while growing the list from **151 to 227**. Adding an address-only search after that found **two** extra links, at **271**. Neither has been tested on the full pool.

## 9. Deciding which candidates match *(Notebook)*

Three ways of deciding were compared on **the same shortlists** for **20,000 random training businesses**, no-match businesses included. The shortlists came from shared rare words (top 30 per source) and found only **66%** of true links. That caps every method, so compare the methods with each other, not with the search results above.

| Method | Score | Precision | Recall | No-match businesses right |
| --- | ---: | ---: | ---: | ---: |
| Hand rule: (name overlap ≥ 90 and address overlap ≥ 80) or (identical cleaned name and same house number) | 0.563 | 79.1% | 49.4% | 51.4% |
| Best single threshold: address word overlap ≥ 97 | 0.569 | 83.5% | 47.9% | 73.7% |
| **Gradient boosting** on 18 simple features (similarity scores, number checks, shortlist score), 5-fold, grouped by business | **0.713** | **96.8%** | 60.8% | 90.9% |
| Best possible with these shortlists | 0.756 | 100% | 66.0% | 100% |

Gradient boosting reached **94%** of the best possible score; the hand rule and best threshold reached about **75%**. Its threshold was tuned on the same predictions it was scored on, so 0.713 is slightly optimistic.

How well each single score separates true matches from wrong shortlisted records (separation score / AUC):

| Score | Separation |
| --- | ---: |
| Address word overlap | **0.966** |
| Cleaned name, whole-text or word-order-free | 0.956 |
| Address partial match | 0.950 |
| Cleaned name, word overlap / Jaro-Winkler / partial | 0.945–0.946 |
| Name with only lowercasing (no other cleaning) | 0.907 |
| Share of address numbers in common | 0.839 |
| Same house number | 0.734 |

**What it means:** a learned model clearly beats hand-written thresholds, and its high precision (97%) suits the score. Cleaning adds real signal (0.956 vs 0.907 for names). House-number agreement only helps in combination with other signals.

## 10. Do embeddings help with Indian script? *(Notebook)*

Setup:
- **Queries:** **2,000 Indian businesses** that have matches.
- **Pool:** all their true records plus **50,000 random Indian S2/S3 records**, about 57,000 in total.
- **Measure:** whether the correct record appears in the top 10 for its business.
- **Hard:** the cleaned names have word overlap below 80.

| Method | All true links | Indian-script names | Hard + Indian-script |
| --- | ---: | ---: | ---: |
| Letter-group text search, raw name | 69.1% | 0.6% | **0.0%** |
| Letter-group text search, cleaned and converted name | 70.5% | 4.8% | **0.0%** |
| Multilingual embeddings (`multilingual-e5-small`, MIT, 118M parameters), raw name | 69.2% | 6.4% | **4.5%** |
| Same embeddings, raw name + address | 95.9% | 92.3% | **91.7%** |

- On names alone, **neither method handles Indian script**: embeddings find 4.5% of the hard Indian-script links.
- **Name + address embeddings find 92%**, but text search on name + address was not run in this test, so most of that gain probably comes from the address rather than the embeddings.
- Codex's full-pool text search on name + address found 85.84% of Indian-script links against a much larger pool, so the two numbers can't be compared directly.

**What it means:** embeddings are not justified yet. Try the learned dictionary and name + address text search first, then test embeddings only on the links that are still missed.

## 11. Duplicates inside S2 and S3 *(Notebook)*

Most businesses have several S2/S3 records, so S2 and S3 contain their own duplicates. From a sample of **300,000 businesses**:

- Two records of the same business are **not** more alike than each is to S1 on names: 60% of same-source pairs score ≥ 90, against 74% for S1-to-record. Their addresses usually agree, though (83–84% score ≥ 90).
- Of **181,090** true links with hard names (word overlap < 80), **137,400 (75.9%)** have another record of the same business that is easy (name ≥ 90 against S1) **and** similar to the hard record (name or address ≥ 90).
- Records with the **same cleaned name and address in one source** belong to the same business **97–98%** of the time (S2 97.8%, S3 97.1%). About 2–2.6% belong to different businesses, and 0.4–0.5% include a distractor.

**What it means:** matching the easy record first and then pulling in its close duplicates could recover most hard links. It needs testing. This check used the labels to group records, so a real pipeline must find duplicates without them, and every pulled-in record risks precision.

## 12. France

France appears **only in test**, so nothing French can be checked against answers. These observations come from the unlabeled test files. *(Fable, Notebook)*

- **Names:** accents are common, and company endings include `SARL`, `SAS`, `SASU`, `EURL` and `SCI`.
- **Address forms:** `r` or `r.` for `rue`, `av` for `avenue`, `All.` for `allée`, the number marker `n°`, and `bis`/`ter` after a number.
- **Damaged words:** some lose their first letter (`oulevard`, `ordeaux`). Others have it split off with a space (`47 B Oulevard Jean Moulin`).
- **Few distinct places:** only about **20 cities and 3 regions**, so place words tell businesses apart poorly. Street details and numbers matter more. *(Fable)*

Measured share per source *(Notebook)*:

| Pattern | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| Region name (`Hauts-de-France`, `Nouvelle-Aquitaine`…) | **100%** | 32% | 35% |
| `rue` written out | 66% | 38% | 39% |
| `r` / `r.` for rue | **0%** | 25% | 24% |
| `n°` / `no` number marker | **0%** | 12% | 11% |
| `bis` / `ter` | 5.0% | 3.8% | 3.8% |
| Five-digit postal code | 0.4% | 0.5% | 0.5% |
| Accented name | 16% | 25% | 24% |
| `SARL` / `SAS(U)` / `EURL` | 28% / 24% / 7% | 21% / 21% / 6% | 21% / 21% / 6% |

- S2/S3 often replace the region with a department (`Nord`, `Gironde`, `Loire-Atlantique`). S2 writes about 29% of French addresses in capitals.
- France looks at least as easy to match by name as the other countries: **96.5%** of French S1 businesses have an S2/S3 record with the same simple name key, against **91–93%** for the US and India. *(Notebook)*
- Among likely French pairs (same simple name key and postal code), company endings sometimes differ (`SCI` vs `SARL`, `SAS` vs `SARL`). Since these pairs are unlabeled, this is only a hint that a different ending should not reject a match. *(Notebook)*

**What it means:** keep accented text; support these address forms; tolerate small spelling damage; don't rely on regions or postal codes. Good US and India results do not prove France will work as well.

## 13. Memory and speed

- With 50 candidates from each source, all training businesses produce **220,682,100 candidate pairs**. Storing 50 four-byte numbers per pair takes about **44.1 GB** before IDs and text. This is a calculation, not a measurement, and it is more than the 24 GB machine used in the independent audit. *(Codex)*
- The full-pool text search processed the data in pieces. For **1,000 businesses** against all S2/S3 records it took *(Codex)*:

| Resource | Measured |
| --- | ---: |
| Building the index, searching and first scoring | 26.7 minutes |
| Peak memory (all processes) | 2.10 GB |
| Reusable files on disk | 17.87 GB |

**What it means:** the full pool can be searched with little memory, but the time for all ~4 million S1 businesses (training + test) is unknown. Work in batches, reuse cleaned text, control shortlist sizes, and measure speed on a larger batch before a full run.

## 14. What to do next

**Cleaning**

1. **Keep every original field**, and add cleaned versions next to it:
   - For names: lowercase, no accents, converted to English letters, no endings, sorted words, and joined words for website names.
   - For addresses: cleaned text, sorted words, numbers and house number.
2. **Company endings: remove them anywhere in the name**, not only at the end. The list must include:
   - English forms: inc, corp, co, company, llc, pllc, ltd, limited, pvt, private, (p), opc, plc, llp, lp, pc.
   - French forms: sarl, sas, sasu, eurl, sa, sci, snc, cie, "& fils", ets.
   - The converted Indian-script spellings (praivet, limitet, elelpi…).
   
   Also drop `and/the/of`, titles (m/s, mr, dr, smt, sri, shri), and junk (www prefixes, `(ID: …)`, phone numbers, `--`, `<<`, `|`).
3. **Split trade names** at dba / aka / fka / t/a / formerly / "doing business as" (S3 only), and compare each part.
4. **Website names:** remove `www.` and the ending (`.com`), then compare with the joined S1 name.
5. **Addresses:**
   - Standardize abbreviations, mapping `saint` → `st` as well.
   - Drop leading zeros in numbers.
   - Put every state name into one form, handling each source's convention and the 16 Indian-script state names.
   - Remove `null` / `n/a`.
   - Take the house number before sorting words.
   - Compare with overlap scores and "one address contains the other", not equality.
6. **Word splitting** must keep Indian vowel signs (`[\p{L}\p{M}\p{N}]+`).

**Indian-script names** (details in section 5)

7. Build the **learned word dictionary** from training pairs, targeting S1's spelling, inside each training split only. Then add the **fallback** (generic conversion with fixes). Consider an **allowed small model** only if a big gap remains.
8. **Do not use Claude or any outside service to convert records.** Use it only to review training examples and suggest rules.
9. Measure success on held-out Indian-script pairs. Today 6.4% reach similarity 90, and search finds 85.84% of these links.

**Candidate search**

10. Use **weighted text search on name + address** (words and letter groups) as the main search. Search within the same country, with indexes built per country so new countries like France work automatically.
11. Add the learned dictionary, company-ending removal, leading-zero cleanup and country filtering to the search text, then remeasure at the same list size and cost.
12. Add fallback searches for the known gaps:
    - A name-only search when the address is missing.
    - An address-only search for weak or unrelated names.
    
    Keep each only if it adds links at a reasonable list size.
13. Goal: **at least 98% of true links found, overall and in each country**, measured on the full pool and not a small pool with the answers inserted. Don't use plain word keys as the main search.

**Testing honestly**

14. Keep businesses used for testing separate from those used to learn rules, dictionaries or models (for example, 5 groups split by business).
15. **Add distractors when testing:** hide about 20% of the test group's S1 businesses but keep their S2/S3 records. That raises the distractor share from about 26% towards the ~40% expected on test.
16. Include convincing wrong pairs, such as same-name businesses and the Ivenent Metal-type cases.
17. As a stand-in for France, train on the US and test on India (and the reverse). A drop larger than 0.03 means some feature only works for one country.

**Deciding matches**

18. **Train a gradient-boosting model** on the shortlisted pairs. It beat hand rules by a wide margin. Give it:
    - Name and address similarity scores, in several kinds and on the cleaned versions.
    - Rare-word address overlap: street and locality, not city or state.
    - House-number and other-number agreement, and number conflicts.
    - Missing-field flags.
    - Website-name and alias checks.
    - Company-ending differences, as a signal only, never a hard rule.
    - The name + house-number match flag.
    - Search scores and ranks.
    - How a candidate compares with the business's best candidate.
    - How strongly other businesses claim the same record.
19. **Allow several correct records per source.** Keeping only the best per source caps the score at 0.857.
20. **Lean towards precision** when choosing thresholds: one wrong extra record costs more than one missed.
21. **Test "one owner per record":** when two businesses claim the same S2/S3 record, give it only to the stronger claim.
22. **Test duplicate expansion:** after accepting a record, also accept same-source records with the same cleaned name and address, or very high similarity, if their own model score is reasonable.
23. Test a separate "does this business have any match?" check for no-match businesses. Keep it only if it improves the score.

**Scale and models**

24. Measure speed on a larger batch (tens of thousands of businesses) before running all ~4 million. Process in batches.
25. Any model used on the data must be MIT or Apache 2.0 and at most 8B parameters; record its licence and size. Try embeddings or re-rankers only on the specific links that simpler steps still miss.

## 15. How much to trust each number

| Evidence | What was examined |
| --- | --- |
| Independent structure and label audit *(Codex)* | All 12,527,040 training rows and all 7,638,365 true links. |
| Independent name/address checks *(Codex)* | 69,721 true links from 20,000 sampled businesses. |
| Full-pool simple search *(Codex)* | 2,000 businesses against all 10,320,219 S2/S3 training records. |
| Smaller-pool text search *(Codex)* | 1,000 businesses against 223,658 records, with the correct records deliberately included. |
| Full-pool text search *(Codex)* | The same 1,000 businesses against all 10,320,219 records; labels used only to score. |
| Training/test comparison and France *(Fable)* | Diagnostics in the shared experiment log; test has no labels. |
| Cleaning steps, word keys, fixed rules, name sharing *(Notebook)* | All 7,638,365 true links and all 2,206,821 training businesses. |
| Deciding matches *(Notebook)* | 20,000 random training businesses; shortlists found 66% of true links. |
| Embeddings *(Notebook)* | 2,000 Indian businesses against about 57,000 records. |
| Duplicates *(Notebook)* | 300,000 training businesses. |
| France *(Notebook)* | Unlabeled test files. |

How the samples were chosen:
- **Codex's** sampled experiments left out a reserved group of about 20% of businesses; its overall counts include that group.
- **The notebook** did not reserve a group.
- The full-pool text-search run is recorded under `eda/full_sparse_20260925/`, and its verification report passes.
- **None** of these results is a final score, and there is no French accuracy result.

Numbers that need careful wording:

- **Training distractors:** "27%" is a rounded figure. The exact share is **25.99%** (S2 26.64%, S3 25.37%).
- **Test distractors:** "about 40%" assumes the same number of matches per business as training. It is an estimate.
- **Identical names after cleaning** (46.31%, ~49%, 55.8%) and **shared names** (38.34%, 49.5%, 48.8–54.8%) come from different rules and samples. They are not the same measurement.
- **Shared words do not prove a match.** 99.99% of true pairs share a name or address word, and fewer than 0.1% have both name and address similarity below 50 *(Fable)*. That does not make 99% recall or score achievable: wrong pairs share words too, and reaching 99.99% with word keys costs about 1.65 million comparisons per business.
- **Finding a record is not matching it.** 99.75% (small pool), 98.36% (full pool, combined list) and 97.80% (full pool, 50 per source) are search results. The final system must also reject wrong candidates and handle no-match businesses.
- **The notebook's 20,000-business scores** are capped by a weak shortlist (66%). Compare the methods with each other only.
- **The embedding test** used a small pool and did not include name + address text search, so it cannot show embeddings beat text search.
- **"Transliteration, not translation"** is based on the most frequent Indian-script words. A few translated names may still exist.

## Source files

- [Fable's detailed diagnostics and the shared experiment log](D:/PROJECTS/amazon_ml_challenge/EXPERIMENTS.md)
- [Codex's independent findings](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/FINDINGS.md) and [experiment scope](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/README.md)
- [Competition task and scoring rules](D:/PROJECTS/amazon_ml_challenge/student_resource/README.md)
- [Full label totals](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/label_summary.tsv), [label integrity checks](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/integrity.tsv), and [verification results](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/verification.txt)
- [Full-pool simple-search results](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/full_blocking_curves.tsv) and [smaller-pool text-search results](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/sparse_probe_curves.tsv)
- [Number comparisons](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/numeric_sensitivity_summary.tsv) and [missing-address counts](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/full_missingness_by_label.tsv)
- [Fable's 30 difficult true pairs and 30 near-misses](D:/PROJECTS/amazon_ml_challenge/eda/out/hard_samples.txt)
- [Codex's 30 difficult true pairs](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/hard_positives_raw.tsv) and [30 convincing wrong pairs](D:/PROJECTS/amazon_ml_challenge/eda/independent_20260925/near_misses_raw.tsv)
- [Full-pool text-search description](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/README.md), [results](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/RESULTS.md), [detailed curves](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/recall_curves.tsv) and [verification](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/verification.txt)
- [Breakdown of full-pool search misses](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/retrieval_error_slices.tsv) and [the 58 missed true pairs](D:/PROJECTS/amazon_ml_challenge/eda/full_sparse_20260925/missed_true_pairs_raw_k50.tsv)
- Notebook: [eda.ipynb](eda_results/eda.ipynb) and its tables in [eda_results/eda_outputs/](eda_results/eda_outputs/):
  - Cleaning steps for [names](eda_results/eda_outputs/04_name_staircase.tsv) and [addresses](eda_results/eda_outputs/04_address_staircase.tsv)
  - [Converted Indian-script words](eda_results/eda_outputs/04_translit_top_tokens.tsv) and [words that differ in true pairs](eda_results/eda_outputs/04_name_token_differences.tsv)
  - [Fixed rules](eda_results/eda_outputs/05_exact_rules.tsv) and [shared names](eda_results/eda_outputs/05_s1_name_collisions.tsv)
  - [Word-key search: found vs cost](eda_results/eda_outputs/06_blocking_recall_cost.tsv)
  - [Rules vs gradient boosting](eda_results/eda_outputs/07_rules_vs_threshold_vs_gbm.tsv) and [single-score separation](eda_results/eda_outputs/07_feature_medians_and_auc.tsv)
  - [Embeddings](eda_results/eda_outputs/08_embeddings_vs_tfidf.tsv)
  - [Duplicates](eda_results/eda_outputs/09_duplicate_rescue.tsv) and [same-owner check](eda_results/eda_outputs/09_exact_duplicates_owner.tsv)
  - [Leak checks](eda_results/eda_outputs/09_leak_checks.tsv)
  - [France patterns](eda_results/eda_outputs/03_france_patterns.tsv)
