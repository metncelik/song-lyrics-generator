#!/bin/bash
for i in {1..10}; do
    python3 /Users/metincelik/Developer/song-lyrics-generator/scrape/scraper.py &
done
wait
