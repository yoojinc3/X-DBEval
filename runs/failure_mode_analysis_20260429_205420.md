# X-DBEval Agent Failure Mode Analysis

**Run date:** 2026-04-29  
**Result:** 3/20 correct (15%) — Tasks 1, 10, 19  
**Tasks with no SQL generated:** 6, 12

---

## Per-Task Breakdown

### Task 1 — Simple ✅ CORRECT
**Pearson correlation, food inspection failure rate vs crime arrest rate**  
`r=0.5181` vs gold `r=0.518` — within tolerance.

---

### Task 2 — Challenging ❌
**Welch's t-test, harsh weather vs domestic violence**  
`t=2.34, p=0.026` vs gold `t=2.27, p=0.113`

The crime SQL used `strftime('%Y-%m', date)` to extract month, but Chicago crime dates are in `M/D/YYYY HH:MM` format (e.g. `1/1/2018 2:46`), not ISO format. SQLite's `strftime` returns NULL for non-ISO dates, so all monthly grouping failed silently and the query returned 268,002 raw rows instead of 12 monthly aggregated rows. The gold SQL uses explicit string manipulation (`SUBSTR`, `PRINTF`) to handle this format. The result generator computed the t-test on incorrectly grouped data, producing a wrong p-value.

**Root cause:** Date format mismatch — `strftime` silently fails on non-ISO dates.

---

### Task 3 — Challenging ❌
**Gender-stratified Welch's t-test, total claims vs mental health burden**  
`female_t=1.234, female_p=0.218, male_t=0.876, male_p=0.381` vs gold `female_t=-0.46, female_p=0.64, male_t=-4.42, male_p=0.0`

Two problems. The Synthea SQL did not `GROUP BY patient` or `SUM(TOTAL)`, returning 12,829 raw claim rows instead of 1,004 patient-level totals. The mental health survey SQL returned 2,958 raw respondent rows instead of 6 aggregated age/gender/mh_rate rows — the burden classification (high vs low per gender using median threshold) was never computed in SQL. The result generator received unstructured raw data and computed a completely wrong t-test.

**Root cause:** Insufficient SQL aggregation — agent delegated too much computation to Python that should have been done in SQL.

---

### Task 4 — Challenging ❌
**Spearman correlation, graduation rate vs adverse outcome rate**  
`spearman_r=0.058, p=0.913` vs gold `spearman_r=-0.06, p=0.913`

P-value matches exactly but the sign of `spearman_r` is flipped. The `college_completion` SQL filtered to only 6 specific school names the agent invented from the question context, instead of using all 3,467 schools with graduation data (gold: `WHERE grad_150_value IS NOT NULL`). The cross-database join then only matched 6 schools, giving a different correlation sign.

**Root cause:** Scope misinterpretation — over-constrained query based on assumed context rather than reading the full table.

---

### Task 5 — Challenging ❌
**Pearson correlation, spending-to-labor ratio vs adverse outcome rate**  
`r=-0.2935` vs gold `r=-0.63`

The selector chose only `student_loan` and `human_resources`, missing `college_completion` entirely — the database that contains `exp_award_value` (institutional spending per award). Without it, the spending-to-labor ratio could not be computed and the agent computed something else entirely.

**Root cause:** Database selection failure — the third required database was dropped, likely because "spending" in the question was not connected to `exp_award_value` in `college_completion` metadata.

---

### Task 6 — Moderate ❌
**Pearson correlation, F1 constructor championship position vs football goals**  
`error: "LLM failed to produce answer or call tools"`

Complete failure before any SQL was generated. Zero tokens consumed, zero API calls made. The agent framework encountered a parsing or validation error and produced no output.

**Root cause:** Infrastructure/framework failure — response was empty or malformed, agent exited before SQL generation.

---

### Task 7 — Simple ❌
**Pearson correlation, ZIP code standard rate vs SAT high achiever rate**  
`r=-0.1703` vs gold `r=-0.1903`

Two issues. The `student_club` ZIP query returned county names with ` County` suffix (e.g. `"Alameda County"`) while `california_schools` returns bare names (e.g. `"Alameda"`). The gold SQL applies `REPLACE(county, ' County', '')` to normalize this. Additionally the SAT query used `satscores.cname` directly (57 rows) instead of joining `schools` on `CDSCode` which the gold uses (48 rows), pulling data from a column with different coverage. The join key mismatch produced incorrect county alignment.

**Root cause:** String normalization failure on the cross-database join key.

---

### Task 8 — Moderate ❌
**Pearson correlation + p-value, British F1 constructor points vs EPL home goals**  
`r=0.2263, p=0.5899` vs gold `r=0.2279, p=0.5872`

Very close but outside tolerance. The F1 SQL queried `constructorResults` (race-level points, e.g. 177 for 2008) instead of `constructorStandings` (championship standings points, e.g. 1792 for 2008) — values differ by ~10x. The football SQL returned 3,040 raw match rows instead of 8 season-level averages, with the result generator computing averages in Python. The F1 table mismatch pushed the answer slightly off.

**Root cause:** Wrong table selection — `constructorResults` vs `constructorStandings` produce very different point values.

---

### Task 9 — Challenging ❌
**Two-proportion z-test, carcinogenicity rate vs thrombosis rate**  
`p1=0.333, p2=0.099, z=7.78` vs gold `p1=0.433, p2=0.099, z=11.16`

`p2` matches exactly. `p1` is wrong because the toxicology SQL used `LEFT JOIN` on the atom table, which includes molecules with no nitrogen atoms (has_nitrogen=0). The denominator n1 thus includes all 343 molecules instead of the 187 that actually contain nitrogen. The gold SQL uses `WHERE molecule_id IN (SELECT DISTINCT molecule_id FROM atom WHERE element='n')` to correctly scope the denominator. The wrong n1 produces a wrong p1 and therefore a wrong z-statistic.

**Root cause:** Wrong filter logic — LEFT JOIN semantics vs IN subquery semantics produce different denominators.

---

### Task 10 — Challenging ✅ CORRECT
**Spearman correlation, F1 constructor win rate vs average total goals**  
`spearman_r=0.8` vs gold `spearman_r=0.8` — exact match.

---

### Task 11 — Simple ❌
**Pearson correlation, F1 circuit latitude vs away goals conceded**  
`r=-0.3206` vs gold `r=0.6134`

Sign is completely wrong. The F1 SQL returned 72 raw circuit rows without `GROUP BY country`, so per-country average latitudes were not computed in SQL. The result generator averaged in Python but the subsequent join with 11 football countries used different country name matching logic, producing incorrect cross-database alignment. With unaveraged raw data being matched against aggregated football data, the correlation is computed over a misaligned dataset.

**Root cause:** Missing `GROUP BY` in SQL causing incorrect Python-side aggregation and cross-database misalignment.

---

### Task 12 — Challenging ❌
**Two-proportion z-test, loan delinquency rate vs thrombosis rate**  
`error: "1 validation error for DBSummary... EOF while parsing a value at line 1 column 0"`

Complete failure before any SQL was generated. The metadata extractor returned an empty string for one of the databases (`financial` or `thrombosis_prediction`), causing a Pydantic JSON validation error during DB summary parsing.

**Root cause:** Infrastructure/framework failure — metadata extraction returned empty content, crashing before SQL generation.

---

### Task 13 — Moderate ❌
**Pearson correlation, F1 constructor championship points vs home goals**  
`r=-0.30` vs gold `r=-0.4127`

The F1 SQL is correct (value_comparison: match=true). The football SQL returned 25,979 raw match rows instead of 11 country-level averages — missing `AVG()` and `GROUP BY co.name`. The result generator computed averages in Python but the F1-to-football country name mapping (e.g. "British" → "England", "German" → "Germany") was done without a lookup table, causing incomplete or incorrect matching. Only a partial set of countries aligned, shifting the correlation.

**Root cause:** Missing SQL aggregation + nationality-to-country name mismatch in cross-database join.

---

### Task 14 — Moderate ❌
**Pearson correlation, GDP per capita vs 2008 Olympic medal count**  
`r=0.1919` vs gold `r=0.2608`

Both SQLs returned correct data (value_comparison: match=true for both). The failure is in the Python join — Olympic medals use region names (e.g. "United States") while WDI uses CountryName with different conventions for some entries (e.g. "Congo, Dem. Rep." vs "Democratic Republic of Congo"). The fuzzy name matching dropped many countries silently, reducing the effective join size and skewing the correlation.

**Root cause:** Country name normalization failure in the cross-database join — no lookup table for name reconciliation.

---

### Task 15 — Challenging ❌
**OLS regression slope, GDP per capita → 2012 Olympic medal count**  
`slope=0.000281` vs gold `slope=0.000538`

Same root cause as Task 14 — both SQLs returned correct data but the Python join dropped many countries due to name mismatches between Olympics region names and WDI CountryName values. Fewer matched countries significantly changes the regression slope. Additionally the agent used `games_name = '2012 Summer'` instead of `games_year = 2012 AND season = 'Summer'` — a more fragile string match.

**Root cause:** Same country name normalization failure as Task 14.

---

### Task 16 — Challenging ❌
**OLS multiple regression, GDP + population → 2012 Olympic medal count**  
`beta_0=4.10, beta_1=0.0009, beta_2=6e-8` vs gold `beta_0=5.55, beta_1=0.000621, beta_2=7.29e-8`

Both GDP and population were fetched in a single query with `IndicatorCode IN ('NY.GDP.PCAP.CD', 'SP.POP.TOTL')`, returning 472 rows in long format (one row per indicator per country). The gold uses two separate queries returning wide format (one row per country). The result generator had to pivot the long table in Python, and the pivot introduced errors or mismatched rows, producing wrong regression coefficients.

**Root cause:** Wrong SQL structure — long format output requires Python pivoting that introduced errors; gold uses two separate wide-format queries.

---

### Task 17 — Simple ❌
**Pearson correlation, crime count per weekday vs average tweet sentiment**  
`r=0.0` vs gold `r=0.419`

The crime SQL used `strftime` on `M/D/YYYY` formatted dates — same failure as Task 2. All weekday extraction returned NULL, so no grouping occurred and 268,002 raw rows were returned instead of 7 weekday-aggregated counts. The social media SQL also returned 99,900 raw rows instead of 7 per-weekday averages. The result generator received unstructured data from both databases and returned 0.0.

**Root cause:** Same date format mismatch as Task 2, plus missing aggregation on the social media side.

---

### Task 18 — Moderate ❌
**Spearman correlation, NBA players per country vs GDP per capita**  
`spearman_r=0.0249` vs gold `spearman_r=0.0474`

Both SQLs returned correct data (value_comparison: match=true). The failure is in the Python join — NBA players use IOC 3-letter country codes (e.g. "USA", "FRA", "YUG") while WDI uses its own country codes. Many codes diverge, and historical codes like "YUG" (Yugoslavia, 23 players) have no WDI equivalent. The result generator joined without a lookup table, silently dropping unmatched countries and skewing the correlation.

**Root cause:** IOC-to-WDI country code mismatch in cross-database join — no code translation table.

---

### Task 19 — Moderate ✅ CORRECT
**Pearson correlation, NBA players per country vs 2000 Olympic medals**  
`r=0.5982` vs gold `r=0.5893` — within tolerance.

---

### Task 20 — Moderate ❌
**Welch's t-test, all-time Olympic medals by income group (High vs Low)**  
`t=4.093, p=0.0001` vs gold `t=4.017, p=0.0002`

Close but outside tolerance. The WDI SQL pre-filtered to only High, Low, and Lower-middle income groups, returning 161 rows instead of the gold's 214 (all income groups). This excluded Upper-middle income countries entirely. The gold returns all income groups and lets Python classify. With an incomplete country set, the t-test operated on different group compositions, shifting both t and p slightly.

**Root cause:** Over-filtering in SQL — pre-filtering income groups instead of returning all and classifying in Python.

---

## Failure Mode Summary

| Failure Mode | Tasks | Count |
|---|---|---|
| Missing SQL aggregation (raw rows instead of grouped) | 2, 3, 8, 11, 13, 17 | 6 |
| Country/entity name normalization in cross-DB join | 14, 15, 18 | 3 |
| Wrong table or column selected | 4, 8, 9 | 3 |
| Date format mismatch (`strftime` fails on non-ISO dates) | 2, 17 | 2 |
| Infrastructure/framework failure | 6, 12 | 2 |
| Database selection missed a required DB | 5 | 1 |
| Over-filtering in SQL | 20 | 1 |
| Wrong SQL structure (long vs wide format) | 16 | 1 |

---

## Key Observations

**Missing SQL aggregation is the dominant failure** (6 tasks). The SQL generator repeatedly returns raw unaggregated rows when the gold expects pre-grouped results. Despite the prompt instruction to "never return raw individual rows," the agent violates this for complex multi-step aggregations. This is most visible in tasks involving date-based grouping, per-patient aggregation, and per-country averages.

**Cross-database entity alignment has no infrastructure** (3+ tasks). Country names, NOC codes, IOC codes, and nationality strings are inconsistent across databases, and the result generator has no lookup table to reconcile them. Unmatched entities are silently dropped, producing wrong correlation inputs. This affects tasks 7, 13, 14, 15, and 18 in varying degrees.

**Date format handling is a recurring blind spot** (tasks 2 and 17). Chicago crime dates use `M/D/YYYY HH:MM` format. The agent consistently uses `strftime` which requires ISO format, silently returning NULL for all rows. The gold SQL handles this with explicit string parsing (`SUBSTR`, `INSTR`, `PRINTF`). The agent needs either a schema-level hint about date formats or a more defensive SQL generation prompt.

**Two tasks failed before SQL generation** (tasks 6 and 12) due to framework errors unrelated to agent reasoning. These should be fixed at the infrastructure level before drawing conclusions about agent performance on those tasks.
