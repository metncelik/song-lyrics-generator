# Song Lyrics Generator

A tool to generate song lyrics using a fine-tuned GPT-2 model trained on lyrics scraped from Genius.

## Features

- **Web Scraping**: Automated lyrics collection from Genius API
- **Model Training**: Fine-tuning GPT-2 models on lyrics
- **Text Generation**: Generate new song lyrics with customizable parameters
- **Web Interface**: Interactive Gradio demo for easy lyrics generation
- **Model Evaluation**: Perplexity calculation and model performance metrics
- **Data Export**: Convert database to CSV format for analysis

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
python scrape/scrape.py
```

The scraper will:
1. Fetch all unused queries from the database
2. Search for songs matching each query on Genius
3. Download and parse the lyrics for songs
4. Save the lyrics to the database
5. Mark the query as used

Alternatively, use the shell script for parallel scraping:
```bash
./scrape/run.sh
```

### Training the Model

Configure training parameters in `training_config.py`, then train a GPT-2 model on the collected lyrics:
```bash
python train.py
```

This will:
1. Fetch lyrics from the database
2. Process and tokenize the data
3. Fine-tune a GPT-2 model on the lyrics
4. Save the trained model to the checkpoints directory

### Generating Lyrics

#### Command Line Interface
Generate new song lyrics using the trained model:
```bash
python inference.py -generate "<prompt>" <model_path>
```

Example:
```bash
python inference.py -generate "aşk" metncelik/tr-lyrics-generator-cosmos-gpt2-large
```

Fine-tuned models are available at [Hugging Face](https://huggingface.co/metncelik/tr-lyrics-generator-cosmos-gpt2-large) and [Hugging Face](https://huggingface.co/metncelik/tr-lyrics-generator-gpt2-uncased).

#### Web Interface
Launch the interactive Gradio demo:
```bash
python demo.py
```

For public sharing:
```bash
python demo.py --share
```

The web interface allows you to:
- Select from available trained models
- Input custom prompts
- Adjust generation parameters (temperature, top-k, top-p, max length)
- Generate and copy lyrics

### Model Evaluation

Evaluate model performance using perplexity metrics:
```bash
python eval.py <model_path>
```

### Data Export

Convert the database to CSV format:
```bash
python db_to_csv.py
```

This creates a `songs.csv` file in the `dataset` directory.

## Project Structure

```
├── scrape/                 # Web scraping components
│   ├── scrape.py          # Main scraper script
│   ├── add_queries.py     # Query management
│   ├── new_queries.txt    # Search queries
│   └── run.sh            # Parallel scraping script
├── database/              # Database components
│   ├── client.py         # Database operations
│   ├── database.db       # SQLite database
│   └── queries.sql       # Database schema
├── checkpoints/           # Trained model checkpoints
├── dataset/              # Exported data
│   └── songs.csv         # CSV export of lyrics
├── tokenizer/            # Saved tokenizer
├── train.py              # Model training script
├── training_config.py    # Training configuration
├── inference.py          # Text generation script
├── demo.py              # Gradio web interface
├── eval.py              # Model evaluation
├── utils.py             # Data processing utilities
├── db_to_csv.py         # Database export utility
└── requirements.txt     # Python dependencies
```

## Configuration

Training parameters can be customized in `training_config.py`:
- Model selection (default: `ytu-ce-cosmos/turkish-gpt2-large`)
- Learning rate, batch size, epochs
- Evaluation and logging intervals
- Hardware optimization settings

## Database Structure

The database has two tables:
- `songs`: Stores song title, artist name, and lyrics
- `queries`: Stores search queries and tracks which ones have been used

## Dependencies

- `transformers`: Hugging Face transformers library
- `torch`: PyTorch for model training
- `datasets`: Dataset processing
- `gradio`: Web interface
- `requests`: HTTP requests for API calls
- `beautifulsoup4`: HTML parsing
- `python-dotenv`: Environment variable management
- `accelerate`: Training acceleration
- `numpy`: Numerical operations 