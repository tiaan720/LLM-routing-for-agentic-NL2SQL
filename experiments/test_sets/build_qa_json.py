#!/usr/bin/env python3
"""
Build a comprehensive QA JSON file from CSV test datasets.
This script extracts questions and queries from CSV files, executes the SQL queries
to get the answers, and creates a unified JSON structure.

Combines functionality to handle both simple queries (without templating) and
complex templated queries with intelligent field extraction.
"""

import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

dotenv.load_dotenv()


class DatabaseExecutor:
    """Handle database connections and query execution."""

    def __init__(self):
        self.connections = {}

    def get_postgres_connection(self, database: str):
        """Get or create a PostgreSQL connection for a specific database."""
        if database not in self.connections:
            try:
                self.connections[database] = psycopg2.connect(
                    host="localhost",
                    port=5432,
                    database=database,
                    user="postgres",
                    password="postgres",
                    cursor_factory=RealDictCursor,
                )
            except psycopg2.Error as e:
                print(f"Failed to connect to database {database}: {e}")
                return None
        return self.connections[database]

    def execute_query(
        self, database: str, query: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute a SQL query and return the results."""
        conn = self.get_postgres_connection(database)
        if not conn:
            return None

        try:
            with conn.cursor() as cursor:
                cursor.execute(query)

                # For SELECT queries (including CTEs that start with WITH), fetch results
                query_upper = query.strip().upper()
                if query_upper.startswith("SELECT") or query_upper.startswith("WITH"):
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                else:
                    # For other queries, just return success indicator
                    conn.commit()
                    return [{"status": "success"}]

        except Exception as e:
            # Don't print detailed error info to keep output clean for simple queries
            conn.rollback()
            return None

    def close_connections(self):
        """Close all database connections."""
        for conn in self.connections.values():
            if conn:
                conn.close()


def is_simple_query(query: str) -> bool:
    """Check if query is simple enough to execute without complex templating."""
    # Avoid queries with curly braces or complex templating
    if "{" in query or "}" in query:
        return False
    return True


def clean_query(query: str) -> str:
    """Clean and prepare SQL query for execution."""

    # Handle curly braces with content - these contain field lists
    def replace_field_list(match):
        content = match.group(1)
        # Split on comma and clean up field names
        fields = [field.strip() for field in content.split(",") if field.strip()]
        return ", ".join(fields)

    # Replace {field1, field2} with field1, field2
    query = re.sub(r"\{([^}]+)\}", replace_field_list, query)

    # For GROUP BY clauses with empty braces, we need to identify the fields to group by
    # This is more complex, so let's use a simpler approach for now
    if "GROUP BY {}" in query or "GROUP BY *" in query:
        # Try to extract the fields from the SELECT clause
        select_match = re.search(
            r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE | re.DOTALL
        )
        if select_match:
            select_fields = select_match.group(1)
            # Extract field names (simple heuristic)
            fields = []
            for field in select_fields.split(","):
                field = field.strip()
                # Skip aggregate functions and aliases
                if not re.search(
                    r"(COUNT|SUM|AVG|MAX|MIN|COALESCE)\s*\(", field, re.IGNORECASE
                ):
                    if " AS " in field.upper():
                        # Take the part before AS
                        field = field.split(" AS ")[0].strip()
                    if "." in field:
                        # Keep table.field format
                        fields.append(field)
                    elif field != "*" and not field.upper().startswith("DISTINCT"):
                        fields.append(field)

            if fields:
                query = query.replace("GROUP BY {}", f'GROUP BY {", ".join(fields)}')
                query = query.replace("GROUP BY *", f'GROUP BY {", ".join(fields)}')
            else:
                # Remove the GROUP BY clause entirely if we can't determine fields
                query = re.sub(r"GROUP BY\s*[\{\}*]*", "", query)

    # Remove any remaining empty braces
    query = re.sub(r"\{\}", "", query)

    # Clean up multiple spaces and commas
    query = re.sub(r",\s*,", ",", query)  # Remove double commas
    query = re.sub(r"SELECT\s*,", "SELECT", query)  # Fix SELECT ,
    query = re.sub(r"GROUP BY\s*$", "", query)  # Remove trailing GROUP BY
    query = re.sub(r"GROUP BY\s*ORDER", "ORDER", query)  # Fix GROUP BY ORDER
    query = re.sub(r"GROUP BY\s*HAVING", "HAVING", query)  # Fix GROUP BY HAVING

    # Handle multiple queries separated by semicolons - take the first valid one
    queries = [q.strip() for q in query.split(";") if q.strip()]
    if queries:
        # Return the first non-empty query
        for q in queries:
            if q and not q.startswith("--"):  # Skip comments
                return q.strip()

    return query.strip()


def extract_multiple_queries(query_text: str) -> List[str]:
    """Extract multiple SQL queries from a single field."""
    # Some entries have multiple queries separated by semicolons
    queries = []
    for query in query_text.split(";"):
        query = query.strip()
        query_upper = query.upper()
        if query and (
            query_upper.startswith("SELECT") or query_upper.startswith("WITH")
        ):
            queries.append(clean_query(query))

    return queries if queries else [clean_query(query_text)]


def format_answer(results: List[Dict[str, Any]]) -> str:
    """Format query results into a readable answer string."""
    if not results:
        return "No results"

    if len(results) == 1:
        # Single result - format as simple value(s)
        result = results[0]
        values = list(result.values())

        if len(values) == 1:
            return str(values[0])
        else:
            return ", ".join(str(v) for v in values if v is not None)
    else:
        # Multiple results - format as a summary or list
        if len(results) <= 10:
            # Show all results if few
            formatted_results = []
            for result in results:
                values = [str(v) for v in result.values() if v is not None]
                formatted_results.append(", ".join(values))
            return "; ".join(formatted_results)
        else:
            # Show count if many results
            return f"{len(results)} records returned"


def process_csv_file(
    file_path: str, executor: DatabaseExecutor, use_simple_only: bool = False
) -> List[Dict[str, Any]]:
    """Process a CSV file and extract QA pairs."""
    qa_pairs = []

    print(f"Processing {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            # Skip pagila examples as requested
            if row.get("db_name") == "pagila":
                continue

            question = row.get("question", "").strip()
            query_text = row.get("query", "").strip()
            db_name = row.get("db_name", "").strip()

            if not all([question, query_text, db_name]):
                continue

            # Try both simple and complex queries
            answer = None
            working_query = None

            if use_simple_only:
                # Only process simple queries
                if not is_simple_query(query_text):
                    continue

                # Try to execute the query as-is
                results = executor.execute_query(db_name, query_text)
                if results is not None:
                    answer = format_answer(results)
                    working_query = query_text
            else:
                # Try simple queries first
                if is_simple_query(query_text):
                    results = executor.execute_query(db_name, query_text)
                    if results is not None:
                        answer = format_answer(results)
                        working_query = query_text
                else:
                    # Try to clean complex templated queries
                    queries = extract_multiple_queries(query_text)

                    for query in queries:
                        if not query:
                            continue

                        cleaned_query = clean_query(query)
                        results = executor.execute_query(db_name, cleaned_query)
                        if results is not None:
                            answer = format_answer(results)
                            working_query = cleaned_query
                            break

                    # If cleaning didn't work, try the original query
                    if answer is None:
                        results = executor.execute_query(db_name, query_text)
                        if results is not None:
                            answer = format_answer(results)
                            working_query = query_text

            # Only add if we got a successful result
            if answer is not None:
                qa_pair = {
                    "question": question,
                    "query": working_query,
                    "answer": answer,
                    "dataset": db_name,
                }

                # Add optional fields if they exist
                if "query_category" in row:
                    qa_pair["category"] = row["query_category"]
                if "instructions" in row:
                    qa_pair["instructions"] = row["instructions"]

                qa_pairs.append(qa_pair)

            # Progress indicator
            if (row_idx + 1) % 50 == 0:
                print(f"  Processed {row_idx + 1} rows...")

    return qa_pairs


def main():
    """Main function to build the QA JSON file."""
    # Initialize database executor
    executor = DatabaseExecutor()

    try:
        # Define test set directory
        test_sets_dir = "/workspaces/DS-masters-2025-llm-rooting/experiments/test_sets"

        # Find all CSV files (excluding pagila_sql_qa.json)
        csv_files = [
            os.path.join(test_sets_dir, f)
            for f in os.listdir(test_sets_dir)
            if f.endswith(".csv")
        ]

        print(f"Found CSV files: {csv_files}")

        all_qa_pairs = []

        # Process each CSV file - try both simple and complex queries
        for csv_file in csv_files:
            qa_pairs = process_csv_file(csv_file, executor, use_simple_only=False)
            all_qa_pairs.extend(qa_pairs)
            print(f"Added {len(qa_pairs)} QA pairs from {os.path.basename(csv_file)}")

        # Save the combined QA JSON file
        output_file = os.path.join(test_sets_dir, "comprehensive_qa.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nSuccessfully created {output_file}")
        print(f"Total QA pairs: {len(all_qa_pairs)}")

        # Print summary by dataset
        datasets = {}
        for qa in all_qa_pairs:
            dataset = qa["dataset"]
            datasets[dataset] = datasets.get(dataset, 0) + 1

        print("\nQA pairs by dataset:")
        for dataset, count in datasets.items():
            print(f"  {dataset}: {count}")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up database connections
        executor.close_connections()


if __name__ == "__main__":
    main()
