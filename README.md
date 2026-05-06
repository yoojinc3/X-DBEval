# X-DBEval: Cross-Database Analytical Benchmark

X-DBEval is a benchmark for evaluating LLM agents on cross-domain SQL analytics tasks. Each task requires an agent to (1) identify the relevant databases from a pool of 26, (2) write correct SQL queries across those databases, and (3) compute a final numeric answer — often involving statistical analysis over joined results.

## Benchmark Overview

- **20 tasks** spanning three difficulty levels: `simple`, `moderate`, `challenging`
- **26 SQLite databases** drawn from the [BIRD benchmark](https://bird-bench.github.io/)
- Tasks require selecting 1–4 databases and producing numeric answers (scalars or dicts)
- Answers are scored with per-metric tolerance (e.g., `{"r": 0.05}`)

### Task Format

Each task in `dev/task.json` has the following fields:

```json
{
  "id": 1,
  "db_id": ["chicago_crime", "food_inspection_2"],
  "difficulty": "simple",
  "question": "Across all Chicago wards, compute the Pearson correlation...",
  "evidence": "A food inspection failure is defined as results = 'Fail'.",
  "intermediate_sqls": [
    {"db": "chicago_crime", "sql": "SELECT ward, ..."},
    {"db": "food_inspection_2", "sql": "SELECT ward, ..."}
  ],
  "result": {"r": 0.518},
  "tolerance": {"r": 0.05}
}
```

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (required for all agent/baseline runs)
- An [OpenAI API key](https://platform.openai.com/) (optional, only for GPT model comparisons)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd X-DBEval_dev
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # optional
```

### 3. Download and set up databases

The SQLite databases are not included in the repository. Download them from the BIRD benchmark:

1. Go to [https://bird-bench.github.io/](https://bird-bench.github.io/) and download the training set
2. Unzip `train.zip` and `train_databases.zip`
3. Copy the database folders into `dev/databases/`

Each database must follow the structure:

```
dev/databases/
  chicago_crime/
    chicago_crime.sqlite
  food_inspection_2/
    food_inspection_2.sqlite
  ...
```

The 26 databases used by this benchmark are: `california_schools`, `card_games`, `chicago_crime`, `codebase_community`, `college_completion`, `debit_card_specializing`, `european_football_2`, `financial`, `food_inspection_2`, `formula_1`, `human_resources`, `mental_health_survey`, `olympics`, `professional_basketball`, `regional_sales`, `sales_in_weather`, `shipping`, `shooting`, `social_media`, `student_club`, `student_loan`, `superhero`, `synthea`, `thrombosis_prediction`, `toxicology`, `world_development_indicators`.

---

## Running an Evaluation

### Agent Pipeline (main approach)

The agent pipeline uses three specialized LLM agents in sequence: a database selector, a SQL generator (with retry), and a result generator.

```bash
python src/run_agent.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `dev/task.json` | Path to task file |
| `--db` | `dev/databases` | Path to database directory |
| `--model` | `claude-sonnet-4-6` | Model for `--summarize`/`--reprice` |
| `--output-dir` | `data/agent_<timestamp>` | Where to write results |
| `--ids` | all | Run only specific task IDs (space-separated) |
| `--start` | 1 | Skip tasks with ID below this value |
| `--summarize` | off | Recompute `summary.json` from existing `results.json` |
| `--reprice` | off | Recompute costs using current pricing then rewrite summary |

**Changing models:** Edit the model constants near the top of `src/run_agent.py`:

```python
MODEL_METADATA = "claude-haiku-4-5-20251001"   # DB metadata extraction (cached)
MODEL_SELECTOR = "claude-opus-4-7"              # Agent 1: database selection
MODEL_SQL      = "claude-opus-4-7"              # Agent 2: SQL generation
MODEL_RESULT   = "claude-opus-4-7"              # Agent 3: result computation
```

**Example — run on specific tasks with Sonnet:**

```bash
# Edit MODEL_SELECTOR/SQL/RESULT to claude-sonnet-4-6 in run_agent.py, then:
python src/run_agent.py --ids 1 2 3 --output-dir data/my_run
```

### Zero-Shot Baseline

Single-turn prompting: the model produces database selection, SQL queries, and result code in one response.

```bash
python src/run_zero_shot.py [--model claude-sonnet-4-6] [--output-dir data/my_zero_shot]
```

Supports the same flags as `run_agent.py`. To use GPT models, set `--model gpt-5.4` (requires `OPENAI_API_KEY`).

### Multi-Turn Baseline

Three-turn prompting without the metadata-driven selector or SQL retry loop.

```bash
python src/run_baseline.py [--model claude-sonnet-4-6] [--output-dir data/my_baseline]
```

---

## Output Format

Each run writes three files to `--output-dir`:

### `results.json`

Array of per-task result objects:

```json
{
  "id": 1,
  "difficulty": "simple",
  "question": "...",
  "selected_dbs": ["chicago_crime", "food_inspection_2"],
  "db_selection_correct": true,
  "turn1_sqls": [{"db": "...", "sql": "..."}],
  "sql_retries": 0,
  "execution_results": [{"db": "...", "sql": "...", "result": [...], "truncated": false}],
  "intermediate_sqls_check": {"available": true, "all_matched": true, "details": [...]},
  "model_answer": {"r": 0.52},
  "gold_answer": {"r": 0.518},
  "correct": true,
  "token_usage": {"input_tokens": 5000, "output_tokens": 300, ...},
  "cost_usd": 0.0123,
  "error": null
}
```

### `summary.json`

Aggregated metrics across all tasks:

```json
{
  "total": 20,
  "correct": 8,
  "accuracy": 0.4,
  "db_selection": {"correct": 18, "total": 20, "accuracy": 0.9},
  "execution_success": 19,
  "execution_success_rate": 0.95,
  "intermediate_sqls": {"tasks_with_gold": 20, "all_matched": 12, "match_rate": 0.6},
  "token_usage": {"input_tokens": 150000, "output_tokens": 8000, ...},
  "cost_usd": 1.45,
  "by_difficulty": {
    "simple":      {"total": 8,  "correct": 5, "accuracy": 0.625},
    "moderate":    {"total": 7,  "correct": 2, "accuracy": 0.286},
    "challenging": {"total": 5,  "correct": 1, "accuracy": 0.2}
  }
}
```

### `log.json`

Step-by-step execution trace for each task (selector output, SQL attempts, retry feedback, generated code, final answer). Useful for debugging failures.

---

## Interpreting Results

| Metric | What it measures |
|--------|-----------------|
| **Overall accuracy** | Fraction of tasks where the final answer is within tolerance of the gold answer |
| **DB selection accuracy** | Fraction of tasks where the model chose exactly the correct databases |
| **Execution success rate** | Fraction of tasks where all SQL queries ran without errors |
| **Intermediate SQL match rate** | Fraction of tasks where model SQL results match the gold intermediate SQL results |
| **Truncated tasks** | Tasks where SQL results exceeded 500 rows (may affect answer quality) |

**Scoring:** Answers are compared numerically with per-metric tolerance. For dict answers (e.g., `{"r": 0.52}`), each key is checked independently. A task is marked `correct` only if all keys pass.

**Cost:** Token usage is tracked per task. The PRICING table in each script reflects current API rates; use `--reprice` to recompute costs if rates change.

---

## Utility Scripts

### Summarize multiple runs

```bash
python src/summarize_results.py results/zero_shot_sonnet_trial1 results/agent_opus
# or scan all *_trial1 directories automatically:
python src/summarize_results.py
```

### Aggregate best-of-N trials

```bash
python src/run_trials.py results/sonnet_trial1 results/sonnet_trial2 results/sonnet_trial3 \
    --output results/sonnet_best_of_3.json
```

This selects the best result per task across trials (correct if any trial succeeded).

---

## Project Structure

```
X-DBEval_dev/
├── src/
│   ├── run_agent.py          # Agent pipeline (main evaluation)
│   ├── run_zero_shot.py      # Zero-shot single-turn baseline
│   ├── run_baseline.py       # Multi-turn baseline
│   ├── run_trials.py         # Best-of-N trial aggregation
│   └── summarize_results.py  # Print summaries for multiple runs
├── dev/
│   ├── task.json             # 20 benchmark tasks with gold answers
│   ├── answer/               # Gold answer computation scripts
│   └── databases/            # 26 SQLite databases (download separately)
├── data/                     # Experiment outputs (gitignored)
├── results/                  # Named trial result directories
├── figures/                  # Visualization notebook and plots
├── requirements.txt
└── .env                      # API keys (gitignored)
```

---

## Benchmark Runner (Agent Integration)

Use `run_agent_benchmark.py` to run the agent on `dev/task.json`, compare the generated SQL against the gold SQL, score answer accuracy with the task `result` and `tolerance` fields, and save a JSON report under `cs498/runs/`.

Useful arguments:

- `benchmark_json`: optional task file path. Defaults to `dev/task.json`.
- `--data-dir`: parent data directory to search for SQLite databases.
- `--db-dir`: direct SQLite database root if you already know the exact folder.
- `--agent-dir`: path to the sibling `cs498-dku-agent` checkout.
- `--task-ids`: comma-separated task IDs to run, such as `3,6,17`.
- `--limit`: run only the first N runnable tasks after filtering missing databases.
- `-o / --output`: output filename inside `cs498/runs/`.
- `--verbose / -v`: print agent logging.

Default starting command:

```bash
python run_agent_benchmark.py --data-dir ../data --limit 2 --verbose
```

If you want to target specific tasks instead:

```bash
python run_agent_benchmark.py --data-dir ../data --task-ids 3,6 --verbose
```

---

## Task Generation Pipeline

### Step 1: Idea Brainstorming from Existing BIRD Examples

Browse `train.json` from the BIRD benchmark understand the question style, difficulty level, and how `evidence` is used to resolve ambiguous terms in the question. Use existing examples as inspiration for cross-domain task ideas.

**Example 1 — `chicago_crime`**
```json
{
    "db_id": "chicago_crime",
    "question": "How many crimes had happened in Central Chicago?",
    "evidence": "Central Chicago refers to district_name = 'Central'",
    "SQL": "SELECT COUNT(*) FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T2.district_name = 'Central'"
}
```

**Example 2 — `social_media`**
```json
{
    "db_id": "social_media",
    "question": "Among all the tweets that have a positive sentiment, how many of them are posted on Thursday?",
    "evidence": "positive sentiment refers to Sentiment > 0; posted on Thursday refers to Weekday = 'Thursday'",
    "SQL": "SELECT COUNT(TweetID) FROM twitter WHERE Sentiment > 0 AND Weekday = 'Thursday'"
}
```

---

### Step 2: Schema Extraction

Extract and review the relevant table schemas from each target database (`.sqlite` files and `database_description` files).
You can use `.schema` command in sqlite3 shell.

**`chicago_crime` — `Crime` table**
```sql
CREATE TABLE Crime (
    report_no            INTEGER primary key,
    date                 TEXT,
    block                TEXT,
    iucr_no              TEXT,
    location_description TEXT,
    arrest               TEXT,
    domestic             TEXT,
    beat                 INTEGER,
    district_no          INTEGER,
    ward_no              INTEGER,
    community_area_no    INTEGER,
    fbi_code_no          TEXT,
    latitude             TEXT,
    longitude            TEXT
);
```

**`social_media` — `twitter` + `location` tables (cross-domain join key: `City`)**
```sql
CREATE TABLE location (
    LocationID INTEGER primary key,
    Country    TEXT,
    State      TEXT,
    StateCode  TEXT,
    City       TEXT       -- join key with chicago_crime via City = 'Chicago'
);

CREATE TABLE twitter (
    TweetID      TEXT primary key,
    Sentiment    REAL,     -- negative sentiment: Sentiment < 0
    "text"       TEXT,
    LocationID   INTEGER,
    UserID       TEXT
);
```

---

### Step 3: Database Modification (if needed)

Check whether a cross-domain join is feasible with the existing schema.
If not, modify the database to enable it.

**This case:** `social_media.location.City` can be used to filter tweets from Chicago, so no modification is needed.

**When modification is needed (example):** If the `location` table did not exist in `social_media`, tweet location information would need to be added manually (e.g., by inserting a `City` column into the `twitter` table and populating it) before a cross-domain join becomes possible.

---

### Step 4: Task Definition

Each task is defined as a **cross-domain analytical question** that requires querying two or more databases and combining the results. The task includes:

- A natural language **question** with specific result format
- The **intermediate SQL queries** needed to retrieve data from each database
- The **result/answer** used for evaluation

**Task format:**
```json
[
    {
        "id": 1,
        "question": "Is there a correlation between the daily number of crimes in Chicago and the daily volume of negative-sentiment tweets posted from Chicago? Report the Pearson correlation coefficient.",
        "evidence": "Negative sentiment refers to Sentiment < 0; Chicago tweets are identified via location.City = 'Chicago'; crime date is parsed from Crime.date (YYYY-MM-DD format)",
        "domains": [
            "chicago_crime",
            "social_media"
        ],
        "SQLs": [
            {
                "db_id": "chicago_crime",
                "description": "Count the number of crimes per day",
                "SQL": "SELECT DATE(date) AS day, COUNT(*) AS crime_count FROM Crime GROUP BY DATE(date)"
            },
            {
                "db_id": "social_media",
                "description": "Count the number of negative-sentiment tweets from Chicago per day",
                "SQL": "SELECT T2.Day, COUNT(*) AS neg_tweet_count FROM twitter AS T1 JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.Sentiment < 0 AND T2.City = 'Chicago' GROUP BY T2.Day"
            }
        ],
        "analysis": "Merge the two query results on the day field, then compute the Pearson correlation coefficient between crime_count and neg_tweet_count.",
        "result/answer": 0.8
    }
]
```

### Step 5: Validation

One person generates the task, and the remaining three members independently validate it. Each validator checks the following:

1. **Question clarity** — Is the question specific and unambiguous enough to have exactly one correct answer?
2. **Evidence completeness** — Does the `evidence` field resolve all domain-specific terms, thresholds, and column references used in the question?
3. **SQL correctness** — Do the intermediate SQLs correctly reflect the intent of the question? Are joins, filters, and aggregations accurate?
4. **Cross-domain linkage** — Is the join key between the two databases valid and unambiguous (e.g., same city name format, same date granularity)?
5. **Result format** — Is the expected output clearly defined and reproducible?

If any validator flags an issue, the task is sent back to the author for revision before being finalized.

---

## BIRD Benchmark — Available Domains

### Address & Geography
- `address`, `mondial_geo`, `world`, `world_development_indicators`

### Airlines & Transportation
- `airline`, `trains`, `shipping`, `bike_share_1`

### Arts & Entertainment
- `disney`, `movie`, `movie_3`, `movie_platform`, `movielens`, `movies_4`, `music_platform_2`, `music_tracker`, `simpson_episodes`, `law_episode`, `shakespeare`, `image_and_language`, `language_corpus`

### Books & Publishing
- `authors`, `books`, `book_publishing_company`, `citeseer`, `cookbook`

### Commerce & Retail
- `car_retails`, `cars`, `regional_sales`, `retail_complains`, `retail_world`, `retails`, `sales`, `sales_in_weather`, `superstore`, `works_cycles`

### Crime & Public Safety
- `chicago_crime`, `shooting`

### Education
- `college_completion`, `computer_student`, `cs_semester`, `student_loan`, `university`

### Finance & Crypto
- `coinmarketcap`, `donor`

### Food & Beverage
- `beer_factory`, `craftbeer`, `food_inspection`, `food_inspection_2`, `menu`, `restaurant`

### Health & Medicine
- `genes`, `mental_health_survey`, `synthea`

### Human Resources & Business
- `human_resources`, `software_company`

### Politics & Government
- `legislator`

### Reviews & Social
- `app_store`, `public_review_platform`, `social_media`, `talkingdata`

### Sports
- `european_football_1`, `hockey`, `ice_hockey_draft`, `olympics`, `professional_basketball`, `soccer_2016`

### Technology
- `codebase_comments`, `video_games`
