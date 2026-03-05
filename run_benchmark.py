#!/usr/bin/env python3
"""
Unified Benchmark SQL Query Runner
====================================

Executes SQL queries from a benchmark JSON file against SQLite databases
and saves formatted results to an output text file.

USAGE:
------
  python run_benchmark.py <benchmark_json> <output_file> [options]

POSITIONAL ARGUMENTS:
  benchmark_json      Path to benchmark JSON file (required)
                      Example: cs498/dev_benchmark_gab.json
  
  output_file         Path to output text file to save results (required)
                      Example: results.txt

OPTIONAL ARGUMENTS:
  -d, --database-dir  Path to database directory (default: data/dev_20240627/dev_databases/dev_databases)
                      Use this to specify a custom location for SQLite .sqlite files
  
  -h, --help          Show this help message and exit

EXAMPLES:
---------
  # Basic usage with default database directory:
  python run_benchmark.py cs498/dev_benchmark_gab.json results.txt
  
  # Run with custom database directory using flag:
  python run_benchmark.py cs498/dev_benchmark_gab.json results.txt -d data/dev_20240627/dev_databases/dev_databases
  
  # Run SID benchmark with output in a specific location:
  python run_benchmark.py cs498/dev_benchmark_sid.json output/benchmark_sid_results.txt
  
  # Run with custom data path (long form):
  python run_benchmark.py cs498/dev_benchmark_gab.json results.txt --database-dir /path/to/databases

OUTPUT:
-------
The script will:
  1. Display progress to the console
  2. Execute all SQL queries from the benchmark JSON file
  3. Display formatted results for each query
  4. Save complete results to the specified output file
  5. Show total rows returned for each query result

DATABASE DIRECTORY STRUCTURE:
----------------------------
The database directory should contain subdirectories for each database:
  database_dir/
  ├── formula_1/
  │   └── formula_1.sqlite
  ├── european_football_2/
  │   └── european_football_2.sqlite
  ├── california_schools/
  │   └── california_schools.sqlite
  ├── financial/
  │   └── financial.sqlite
  └── ...other databases...

If the database directory is not found or contains missing .sqlite files,
the script will display an error message indicating which database(s) are missing.
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class BenchmarkRunner:
    """
    Runs SQL benchmark queries from a JSON specification file.
    
    This class manages:
    - Loading benchmark questions from JSON files
    - Connecting to SQLite databases in a specified directory
    - Executing SQL queries and formatting results
    - Writing output to both console and file
    
    The database directory can be customized to point to any location
    containing the SQLite database files.
    """
    
    def __init__(self, benchmark_file: Path, database_dir: Path):
        """
        Initialize the benchmark runner.
        
        Args:
            benchmark_file (Path): Path to the benchmark JSON file
            database_dir (Path): Path to directory containing SQLite database files
                                Each database should be in subdirectory with matching name
                                Example structure:
                                  database_dir/
                                  ├── formula_1/formula_1.sqlite
                                  ├── european_football_2/european_football_2.sqlite
                                  └── ...
        """
        self.benchmark_file = benchmark_file
        self.database_dir = database_dir
        self.connections: Dict[str, sqlite3.Connection] = {}
        self.output_lines: List[str] = []
    
    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Load benchmark questions from JSON file."""
        with open(self.benchmark_file, 'r') as f:
            return json.load(f)
    
    def get_connection(self, db_id: str) -> sqlite3.Connection:
        """
        Get or create a database connection.
        
        Constructs the database file path as: database_dir/{db_id}/{db_id}.sqlite
        Caches connections to avoid reopening the same database multiple times.
        
        Args:
            db_id (str): Database identifier (e.g., 'formula_1', 'european_football_2')
        
        Returns:
            sqlite3.Connection: Connection to the SQLite database
        
        Raises:
            FileNotFoundError: If the database file cannot be found at the expected path
        
        Examples:
            With database_dir = "data/dev_databases":
            - get_connection("formula_1") looks for: data/dev_databases/formula_1/formula_1.sqlite
            - get_connection("california_schools") looks for: data/dev_databases/california_schools/california_schools.sqlite
        """
        if db_id not in self.connections:
            db_path = self.database_dir / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            self.connections[db_id] = conn
        return self.connections[db_id]
    
    def close_connections(self):
        """Close all database connections."""
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()
    
    def execute_query(self, conn: sqlite3.Connection, sql: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        return results
    
    def format_table(self, results: List[Dict[str, Any]], max_rows: int = 50) -> str:
        """Format query results as a readable table."""
        if not results:
            return "  (No results)"
        
        display_results = results[:max_rows]
        keys = list(display_results[0].keys())
        
        # Calculate column widths
        col_widths = {key: min(len(str(key)), 60) for key in keys}
        for result in display_results:
            for key in keys:
                val_len = len(str(result[key]))
                col_widths[key] = min(max(col_widths[key], val_len), 60)
        
        # Build header and separator
        header = "  " + " | ".join(f"{key:<{col_widths[key]}}" for key in keys)
        separator = "  " + "-+-".join("-" * col_widths[key] for key in keys)
        
        # Build rows
        rows = [header, separator]
        for result in display_results:
            row = "  " + " | ".join(f"{str(result[key]):<{col_widths[key]}}" for key in keys)
            rows.append(row)
        
        output = "\n".join(rows)
        
        # Row count
        if len(results) > max_rows:
            output += f"\n  ... (showing {max_rows} of {len(results)} rows)"
        else:
            output += f"\n  (Total: {len(results)} rows)"
        
        return output
    
    def write_output(self, text: str = ""):
        """Write to console and store for file output."""
        print(text)
        self.output_lines.append(text)
    
    def run(self) -> bool:
        """Execute all benchmark queries."""
        try:
            # Load benchmark
            questions = self.load_benchmark()
            
            # Header
            self.write_output("=" * 80)
            self.write_output(f"BENCHMARK QUERY RUNNER")
            self.write_output(f"Benchmark File: {self.benchmark_file}")
            self.write_output(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.write_output("=" * 80)
            self.write_output(f"Loaded {len(questions)} questions\n")
            
            # Process each question
            for question in questions:
                q_id = question.get("id", "?")
                difficulty = question.get("difficulty", "?")
                q_text = question.get("question", "")
                
                self.write_output(f"\n{'=' * 80}")
                self.write_output(f"Question {q_id} ({difficulty.upper()})")
                self.write_output(f"{'=' * 80}")
                self.write_output(q_text)
                self.write_output()
                
                # Execute each SQL query
                sqls = question.get("SQLs", [])
                for idx, sql_item in enumerate(sqls, 1):
                    db_id = sql_item.get("db_id", "?")
                    description = sql_item.get("description", "")
                    sql_query = sql_item.get("SQL", "")
                    
                    self.write_output(f"Query {idx}/{len(sqls)} - Database: {db_id}")
                    self.write_output(f"Description: {description}")
                    self.write_output(f"SQL: {sql_query}")
                    self.write_output()
                    
                    try:
                        conn = self.get_connection(db_id)
                        results = self.execute_query(conn, sql_query)
                        
                        self.write_output("Results:")
                        self.write_output(self.format_table(results))
                        self.write_output()
                        
                    except sqlite3.Error as e:
                        self.write_output(f"ERROR: {e}")
                        self.write_output()
            
            # Footer
            self.write_output(f"\n{'=' * 80}")
            self.write_output("Benchmark execution completed successfully")
            self.write_output("=" * 80)
            
            return True
            
        except Exception as e:
            self.write_output(f"EXECUTION ERROR: {e}")
            return False
        
        finally:
            self.close_connections()
    
    def save_output(self, output_file: Path):
        """Save all output to a file."""
        with open(output_file, 'w') as f:
            f.write("\n".join(self.output_lines))
        print(f"\nOutput saved to: {output_file}")


def main():
    """
    Main entry point.
    
    Parses command-line arguments and executes the benchmark runner.
    Supports both positional arguments and optional flags for flexibility.
    """
    import argparse
    
    # Create argument parser with detailed help
    parser = argparse.ArgumentParser(
        prog='run_benchmark.py',
        description='Execute SQL benchmark queries and save results to file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python run_benchmark.py cs498/dev_benchmark_gab.json results.txt
  python run_benchmark.py cs498/dev_benchmark_gab.json results.txt -d data/custom_databases
  python run_benchmark.py cs498/dev_benchmark_sid.json output/results.txt --database-dir /path/to/db
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'benchmark_json',
        help='Path to benchmark JSON file (e.g., cs498/dev_benchmark_gab.json)'
    )
    parser.add_argument(
        'output_file',
        help='Path to output text file for results (e.g., results.txt)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-d', '--database-dir',
        type=str,
        default='data/dev_20240627/dev_databases/dev_databases',
        help='Path to database directory (default: data/dev_20240627/dev_databases/dev_databases)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    benchmark_file = Path(args.benchmark_json)
    output_file = Path(args.output_file)
    database_dir = Path(args.database_dir)
    
    # Verify inputs
    if not benchmark_file.exists():
        print(f"Error: Benchmark file not found: {benchmark_file}")
        sys.exit(1)
    
    if not database_dir.exists():
        print(f"Error: Database directory not found: {database_dir}")
        print(f"\nTip: Specify a different database directory using -d flag:")
        print(f"  python run_benchmark.py {benchmark_file} {output_file} -d /path/to/databases")
        sys.exit(1)
    
    # Create parent directory for output file if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Display configuration to user
    print(f"Configuration:")
    print(f"  Benchmark file: {benchmark_file.absolute()}")
    print(f"  Database dir:   {database_dir.absolute()}")
    print(f"  Output file:    {output_file.absolute()}")
    print()
    
    # Run benchmark
    runner = BenchmarkRunner(benchmark_file, database_dir)
    success = runner.run()
    
    # Save output
    runner.save_output(output_file)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
