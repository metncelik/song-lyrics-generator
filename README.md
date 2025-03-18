# Song Lyrics Generator

A tool to generate Turkish song lyrics using a fine-tuned GPT-2 model trained on lyrics scraped from Genius.

> **Future Development**: This project will expand to support fine-tuning of various language models using lyrics from all languages, not limited to Turkish songs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get a Genius API token from [here](https://genius.com/api-clients) and add it to your environment variables:
```bash
export GENIUS_ACCESS_TOKEN=your_genius_api_token
```
or create a `.env` file in the `scrape` directory:
```
GENIUS_ACCESS_TOKEN=your_genius_api_token
```

3. Add queries to the database:
First, create a file called `new_queries.txt` in the `scrape` directory and add your queries to it.
Then, run the following command to add the queries to the database:
```bash
python scrape/add_queries.py
```

## Usage

### Scraping Lyrics

Run the scraper:
```bash
python scrape/scraper.py
```

The scraper will:
1. Fetch all unused queries from the database
2. Search for songs matching each query on Genius
3. Download and parse the lyrics for Turkish songs
4. Save the lyrics to the database
5. Mark the query as used

Alternatively, use the shell script:
```bash
./scrape/run.sh
```

### Training the Model

Train a GPT-2 model on the collected lyrics:
```bash
python train/trainer.py
```

This will:
1. Fetch lyrics from the database
2. Process and tokenize the data
3. Fine-tune a GPT-2 model on the lyrics
4. Save the trained model

### Generating Lyrics

Generate new song lyrics using the trained model:
```bash
python inference/inference.py
```

## Database Structure

The database has two tables:
- `lyrics`: Stores song title, artist name, and lyrics
- `queries`: Stores search queries and tracks which ones have been used 