from database .client import DatabaseClient
from utils import process_data

def db_to_csv():
    db_client = DatabaseClient()
    songs = db_client.get_songs()
    processed_songs = [process_data(item) for item in songs]
    with open("dataset/songs.csv", "w") as f:
        f.write(f"lyrics,title,artist\n")
        for song in processed_songs:
            f.write(f"{song['lyrics']},{song['song_title']},{song['artist_name']}\n")
    db_client.close()
    
if __name__ == "__main__":
    db_to_csv()