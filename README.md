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

- A natural language **question** with specific result format (or we can have separate field for this one)
- The **intermediate SQL queries** needed to retrieve data from each database, but won't be used for the evaluation
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


