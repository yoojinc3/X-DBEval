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
- The **intermediate SQL queries** needed to retrieve data from each database (not used for evaluation, but used to validate model SQL)
- The **result** used for evaluation

**Task format:**
```json
[
    {
        "id": 1,
        "difficulty": "simple",
        "question": "Is there a correlation between the daily number of crimes in Chicago and the daily volume of negative-sentiment tweets posted from Chicago? Report the Pearson correlation coefficient.",
        "evidence": "Negative sentiment refers to Sentiment < 0; Chicago tweets are identified via location.City = 'Chicago'; crime date is parsed from Crime.date (YYYY-MM-DD format)",
        "db_id": [
            "chicago_crime",
            "social_media"
        ],
        "intermediate_sqls": [
            {
                "db": "chicago_crime",
                "sql": "SELECT DATE(date) AS day, COUNT(*) AS crime_count FROM Crime GROUP BY DATE(date)"
            },
            {
                "db": "social_media",
                "sql": "SELECT T2.Day, COUNT(*) AS neg_tweet_count FROM twitter AS T1 JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.Sentiment < 0 AND T2.City = 'Chicago' GROUP BY T2.Day"
            }
        ],
        "analysis": "Merge the two query results on the day field, then compute the Pearson correlation coefficient between crime_count and neg_tweet_count.",
        "result": {
            "r": 0.8
        },
        "tolerance": {
            "r": 0.05
        }
    }
]
```

**Field reference:**

| Field | Type | Description |
|---|---|---|
| `id` | int | Unique task identifier |
| `difficulty` | string | `"simple"`, `"intermediate"`, or `"challenging"` |
| `question` | string | Natural language question posed to the model |
| `evidence` | string | Clarifies domain-specific terms, thresholds, and column references |
| `db_id` | string[] | List of database names required to answer the question |
| `intermediate_sqls` | object[] | Gold SQL queries per database (`db` + `sql` keys); used to validate model output |
| `analysis` | string | Steps to derive the final answer from raw query results |
| `result` | object | Expected answer; keys match what the model should return |
| `tolerance` | object | Per-key numeric tolerance for answer comparison (same keys as `result`) |

### Step 5: Validation

One person generates the task, and the remaining three members independently validate it. Each validator checks the following:

1. **Question clarity** — Is the question specific and unambiguous enough to have exactly one correct answer?
2. **Evidence completeness** — Does the `evidence` field resolve all domain-specific terms, thresholds, and column references used in the question?
3. **SQL correctness** — Do the intermediate SQLs correctly reflect the intent of the question? Are joins, filters, and aggregations accurate?
4. **Cross-domain linkage** — Is the join key between the two databases valid and unambiguous (e.g., same city name format, same date granularity)?
5. **Result format** — Is the expected output clearly defined and reproducible?

If any validator flags an issue, the task is sent back to the author for revision before being finalized.

---

# Running the Evaluation

## Prerequisites

```bash
pip install anthropic openai python-dotenv
```

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...         # only needed for GPT models
```

The SQLite databases are not included in the repo (listed in `.gitignore`). Place them at `dev/databases/<db_name>/<db_name>.sqlite`.

---

## Scripts

### `dump_schemas.py` — Extract schemas for offline use

Dumps all database schemas to a single JSON file. Run this once on the machine where the databases live; the output can be used locally by `run_baseline.py`.

```bash
python dump_schemas.py --database-dir /path/to/databases --output schemas.json
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--database-dir` | ✓ | — | Directory containing database subdirectories |
| `--output` | | `schemas.json` | Output JSON file path |

---

### `run_benchmark.py` — Execute gold SQL queries

Runs the pre-defined gold SQL queries from a benchmark file and prints formatted results. Useful for inspecting expected outputs; does not call any LLM.

```bash
python run_benchmark.py dev_benchmark_gab.json results.txt -d /path/to/databases
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `benchmark_json` | ✓ | — | Path to benchmark JSON file |
| `output_file` | ✓ | — | Path to output text file |
| `-d / --database-dir` | | `data/dev_20240627/dev_databases/dev_databases` | Database directory |

---

### `run_baseline.py` — Single-prompt LLM evaluation (root-level)

Sends the merged schema + question to Claude in a single turn, executes the returned SQL, applies statistical post-processing, and scores against gold answers.

```bash
python run_baseline.py dev_benchmark_gab.json results.txt \
  --schema-file schemas.json \
  --database-dir /path/to/databases \
  --model claude-sonnet-4-6
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `benchmark_json` | ✓ | — | Path to benchmark JSON file |
| `output_file` | ✓ | — | Path to output text file |
| `--schema-file` | ✓ | — | Path to `schemas.json` from `dump_schemas.py` |
| `--database-dir` | ✓ | — | Path to directory containing SQLite databases |
| `--model` | | `claude-sonnet-4-5` | Anthropic model ID |

---

### `src/run_agent.py` — Multi-step agentic evaluation

Runs a three-step agentic pipeline: database selection → SQL generation → Python-based result analysis. Supports automatic retries on SQL failure and validates intermediate SQL against gold queries. Outputs `results.json`, `log.json`, and `summary.json` to a timestamped directory.

```bash
python src/run_agent.py \
  --task dev/task.json \
  --db dev/databases \
  --model claude-sonnet-4-6 \
  --output-dir results/my_run
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--task` | | `dev/task.json` | Path to task JSON file |
| `--db` | | `dev/databases` | Path to database directory |
| `--model` | | `claude-sonnet-4-6` | Model ID |
| `--output-dir` | | auto-timestamped | Output directory for results |
| `--ids` | | all | Space-separated list of task IDs to run |
| `--start` | | — | Skip tasks with `id` less than this value |
| `--summarize` | | — | Recompute `summary.json` from existing `results.json` and exit |
| `--reprice` | | — | Recompute `cost_usd` using current pricing and rewrite `summary.json` |

---

### `src/run_baseline.py` — Enhanced single-prompt evaluation

Similar to root `run_baseline.py` but supports both Anthropic and OpenAI models, validates database selection, checks intermediate SQL matching, and reports accuracy by difficulty level.

```bash
python src/run_baseline.py \
  --task dev/task.json \
  --db dev/databases \
  --model claude-haiku-4-5-20251001 \
  --output-dir results/haiku_baseline
```

Arguments are identical to `src/run_agent.py`. Default model is `claude-haiku-4-5-20251001`. For OpenAI models, prefix the model name with `gpt-` (e.g. `--model gpt-4o`).

---

### `src/run_trials.py` — Aggregate best-of-N trial results

Takes multiple trial directories (each containing a `results.json`) and produces an aggregated result that picks the best answer per task across all trials.

```bash
python src/run_trials.py results/trial1 results/trial2 results/trial3 \
  --output results/best_of_3.json
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `trial_dirs` | ✓ | — | One or more paths to trial output directories |
| `--output` | | stdout | Write aggregated JSON to this file |

---

# BIRD Benchmark — Available Domains

## Address & Geography
- `address`
- `mondial_geo`
- `world`
- `world_development_indicators`

## Airlines & Transportation
- `airline`
- `trains`
- `shipping`
- `bike_share_1`

## Arts & Entertainment
- `disney`
- `movie`
- `movie_3`
- `movie_platform`
- `movielens`
- `movies_4`
- `music_platform_2`
- `music_tracker`
- `simpson_episodes`
- `law_episode`
- `shakespeare`
- `image_and_language`
- `language_corpus`

## Books & Publishing
- `authors`
- `books`
- `book_publishing_company`
- `citeseer`
- `cookbook`

## Commerce & Retail
- `car_retails`
- `cars`
- `regional_sales`
- `retail_complains`
- `retail_world`
- `retails`
- `sales`
- `sales_in_weather`
- `superstore`
- `works_cycles`

## Crime & Public Safety
- `chicago_crime`
- `shooting`

## Education
- `college_completion`
- `computer_student`
- `cs_semester`
- `student_loan`
- `university`

## Finance & Crypto
- `coinmarketcap`
- `donor`

## Food & Beverage
- `beer_factory`
- `craftbeer`
- `food_inspection`
- `food_inspection_2`
- `menu`
- `restaurant`

## Health & Medicine
- `genes`
- `mental_health_survey`
- `synthea`

## Human Resources & Business
- `human_resources`
- `software_company`

## Politics & Government
- `legislator`

## Reviews & Social
- `app_store`
- `public_review_platform`
- `social_media`
- `talkingdata`

## Sports
- `european_football_1`
- `hockey`
- `ice_hockey_draft`
- `olympics`
- `professional_basketball`
- `soccer_2016`

## Technology
- `codebase_comments`
- `video_games`


