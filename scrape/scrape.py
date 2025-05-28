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


def get_free_proxies():
    url = "https://www.sslproxies.org/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    proxies = []
    for row in soup.find("table", {"class": "table-striped"}).find_all("tr")[1:]:
        tds = row.find_all("td")
        if len(tds) > 1:
            ip = tds[0].text
            port = tds[1].text
            proxies.append(f"{ip}:{port}")
    return proxies


def make_request_with_proxy(url, headers=None, params=None, use_proxy=False):
    if not use_proxy:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            return response
        except Exception as e:
            print(f"Failed without proxy: {e}")
    
    proxies_list = get_free_proxies()
    
    while proxies_list:
        proxy = random.choice(proxies_list)
        proxy_dict = {
            "http": f"http://{proxy}",
            "https": f"http://{proxy}"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, proxies=proxy_dict, timeout=5)
            print(f"Success with proxy: {proxy}")
            return response
        except:
            print(f"Failed with proxy: {proxy}")
            proxies_list.remove(proxy)
    
    raise Exception("No working proxies found")


def get_song_ids(query, use_proxy=False):
    search_url = BASE_URL + '/search'
    params = {'q': f'{query}'}
    response = make_request_with_proxy(search_url, headers=HEADERS, params=params, use_proxy=use_proxy)
    json_response = response.json()
    song_ids = [hit['result']['id']
                for hit in json_response['response']['hits']]
    return song_ids


def get_song_id(query, use_proxy=False):
    song_ids = get_song_ids(query, use_proxy)
    return song_ids[0]


def get_song_details(song_id, use_proxy=False):
    song_url = BASE_URL + '/songs/' + str(song_id)
    response = make_request_with_proxy(song_url, headers=HEADERS, use_proxy=use_proxy)
    json_response = response.json()
    song_title = json_response['response']['song']['title']
    artist_name = json_response['response']['song']['primary_artist']['name']
    lyrics_path = json_response['response']['song']['path']
    language = json_response['response']['song']['language']
    lyrics_url = 'https://genius.com' + lyrics_path
    return song_title, artist_name, lyrics_url, language


def get_lyrics_page(lyrics_url, use_proxy=False):
    response = make_request_with_proxy(lyrics_url, headers=HEADERS, use_proxy=use_proxy)
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
    
    USE_PROXY = False

    for query in queries:
        query_text = query[1]
        print("-------------------")
        print("? Query: ", query_text)
        print("-------------------")
        
        song_ids = get_song_ids(query_text, USE_PROXY)
        for song_id in song_ids:
            try:
                song_title, artist_name, lyrics_url, language = get_song_details(song_id, USE_PROXY)
                print("*", song_title, " - ", artist_name)
                
                if language != 'tr' or 'Türkçe Çeviri' in song_title:
                    raise Exception("Song is not turkish")

                if db_client.is_song_in_db(song_title, artist_name):
                    raise Exception("Song already exists in db")
                
                html_page = get_lyrics_page(lyrics_url, USE_PROXY)
                parsed_lyrics = parse_lyrics(html_page)
                
                db_client.save_lyrics(song_title, artist_name, parsed_lyrics)
                print("+ Song added to db.")
                
            except Exception as e:
                print("X Error: ", e)
                
        db_client.mark_query_as_used(query_text)
    db_client.close()
        
