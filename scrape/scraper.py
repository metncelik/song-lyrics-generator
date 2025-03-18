import os
import re
import random
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from database.client import DatabaseClient

db_client = DatabaseClient()

load_dotenv()

ACCESS_TOKEN = os.getenv('GENIUS_ACCESS_TOKEN')
BASE_URL = 'https://api.genius.com'

HEADERS = {'Authorization': f'Bearer {ACCESS_TOKEN}'}


def get_song_ids(query):
    search_url = BASE_URL + '/search'
    params = {'q': f'{query}'}
    response = requests.get(search_url, headers=HEADERS, params=params)
    json_response = response.json()
    song_ids = [hit['result']['id']
                for hit in json_response['response']['hits']]
    return song_ids


def get_song_id(query):
    song_ids = get_song_ids(query)
    return song_ids[0]


def get_song_details(song_id):
    song_url = BASE_URL + '/songs/' + str(song_id)
    response = requests.get(song_url, headers=HEADERS)
    json_response = response.json()
    song_title = json_response['response']['song']['title']
    artist_name = json_response['response']['song']['primary_artist']['name']
    lyrics_path = json_response['response']['song']['path']
    language = json_response['response']['song']['language']
    lyrics_url = 'https://genius.com' + lyrics_path
    return song_title, artist_name, lyrics_url, language


def get_lyrics_page(lyrics_url):
    response = requests.get(lyrics_url, headers=HEADERS)
    return response.content


def parse_lyrics(html_page):
    soup = BeautifulSoup(html_page, 'html.parser')
    lyrics_container = soup.select_one('[class^="Lyrics__Container"]')

    if not lyrics_container:
        raise ValueError("Lyrics not found")

    lyrics_text = ""
    for element in lyrics_container.contents:
        if element.name == 'br':
            lyrics_text += '\n'
        elif isinstance(element, str):
            lyrics_text += element
        elif element.name == 'a':
            for span in element.find_all('span', class_=lambda c: c and 'Highlight' in c):
                lyrics_text += span.get_text(separator='\n')

    lyrics_text = re.sub(r'\[[^\]]*\]', '', lyrics_text)  # []
    lyrics_text = re.sub(r'\([^()]*\)', '', lyrics_text)
    lyrics_text = re.sub(r'\n+', '\n', lyrics_text)
    lyrics_text = re.sub(r' +', ' ', lyrics_text)
    lyrics_text = lyrics_text.replace("\n\n", "\n")
    lyrics_text = lyrics_text.strip()

    return lyrics_text


if __name__ == '__main__':
    queries = db_client.get_unused_queries()
    
    random.shuffle(queries)

    for query in queries:
        query_text = query[1]
        print("-------------------")
        print("? Query: ", query_text)
        print("-------------------")
        
        eval_song_count = 0
        eval_in_db_count = 0
        eval_not_tr_count = 0
        
        song_ids = get_song_ids(query_text)
        for song_id in song_ids:
            eval_song_count += 1
            try:
                song_title, artist_name, lyrics_url, language = get_song_details(song_id)
                print("*", song_title, " - ", artist_name)
                
                if language != 'tr' or 'Türkçe Çeviri' in song_title:
                    eval_not_tr_count += 1
                    raise Exception("Song is not turkish")

                if db_client.is_song_in_db(song_title, artist_name):
                    eval_in_db_count += 1
                    raise Exception("Song already exists in db")
                
                html_page = get_lyrics_page(lyrics_url)
                parsed_lyrics = parse_lyrics(html_page)
                
                db_client.save_lyrics(song_title, artist_name, parsed_lyrics)
                print("+ Song added to db.")
                
            except Exception as e:
                print("X Error: ", e)
            print(f"% error_rate: {(eval_in_db_count+eval_not_tr_count)/eval_song_count}. in_db rate: {eval_in_db_count/eval_song_count}")
                
        db_client.mark_query_as_used(query_text)
    db_client.close()
        
