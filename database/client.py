import sqlite3
import os


class DatabaseClient:
    def __init__(self, db_path=None):
        if db_path is None:
            db_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(db_dir, 'database.db')

        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.initialize_tables()

    def initialize_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, song_title TEXT, artist_name TEXT, lyrics TEXT)
        """)
        self.connection.commit()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT UNIQUE, is_used BOOLEAN DEFAULT FALSE)
        """) 
        self.connection.commit()

    def is_song_in_db(self, song_title, artist_name):
        self.cursor.execute(
            """
            SELECT * from songs 
            WHERE song_title = ? 
            AND artist_name = ?
            """, (song_title, artist_name))
        return self.cursor.fetchone() is not None

    def save_lyrics(self, song_title, artist_name, lyrics):
        self.cursor.execute(
            "INSERT INTO songs (song_title, artist_name, lyrics) VALUES (?, ?, ?)",
            (song_title, artist_name, lyrics))
        self.connection.commit()

    def get_unused_queries(self):
        self.cursor.execute(
            "SELECT * FROM queries WHERE is_used = FALSE ORDER BY query_text"
        )
        return self.cursor.fetchall()

    def mark_query_as_used(self, query_text):
        self.cursor.execute(
            "UPDATE queries SET is_used = TRUE WHERE query_text = ?",
            (query_text,))
        self.connection.commit()

    def get_songs(self, limit = None):
        query = "SELECT lyrics, song_title, artist_name FROM songs"
        if limit is not None:
            query += f" LIMIT ?"
            self.cursor.execute(
                query,
                (limit, )
            )
        else:
            self.cursor.execute(query)
        
        return self.cursor.fetchall()

    def close(self):
        self.connection.close()
