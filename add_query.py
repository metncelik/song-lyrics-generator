#!/usr/bin/env python3
"""
Helper script to add queries to the database.
Usage: python add_query.py "your search query"
"""

import sys
from database.client import DatabaseClient

def add_query(query_text):
    """Add a query to the database"""
    db = DatabaseClient()
    try:
        db.cursor.execute("INSERT INTO queries (query_text) VALUES (?)", (query_text,))
        db.connection.commit()
        print(f"Added query: {query_text}")
    except Exception as e:
        print(f"Error adding query: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_query.py \"your search query\"")
        sys.exit(1)
    
    query_text = sys.argv[1]
    add_query(query_text) 