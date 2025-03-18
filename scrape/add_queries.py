import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.client import DatabaseClient

client = DatabaseClient()

file = open("scrape/new_queries.txt", "r")
new_queries = file.read().splitlines()

added_query_count = 0
for query in new_queries:
    try:
        client.cursor.execute("SELECT * FROM queries WHERE query_text = ?", (query,))
        if client.cursor.fetchone() is None:
            client.cursor.execute(
                "INSERT INTO queries (query_text) VALUES (?)", 
                (query,)
            )
            client.connection.commit()
            print("Query added: ", query)
            added_query_count += 1
        else:
            print("Query already exists: ", query)
    except Exception as e:
        print("Error: ", e)

print(added_query_count, " new queries added.")

client.close()