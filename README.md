# Song Lyrics Generator

A tool to generate song lyrics using lyrics scraped from Genius.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your Genius API token:
```
GENIUS_ACCESS_TOKEN=your_genius_api_token
```

3. Add queries to the database:
First, create a file called `new_queries.txt` in the `scraper` directory and add your queries to it.
Then, run the following command to add the queries to the database:
```bash
python scraper/add_queries.py
```

## Usage

Run the scraper:
```bash
python scraper/main.py
```

The scraper will:
1. Fetch all unused queries from the database
2. Search for songs matching each query on Genius
3. Download and parse the lyrics for songs
4. Save the lyrics to the database
5. Mark the query as used

## Song Generation

*Coming soon: The ability to generate new songs based on the collected lyrics.*

## Database Structure

The database has two tables:
- `lyrics`: Stores song title, artist name, and lyrics
- `queries`: Stores search queries and tracks which ones have been used 