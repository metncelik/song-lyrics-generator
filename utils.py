import re

def process_lyrics(lyrics):
    unwanted = [
        "\"",
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "-",
        "_",
        "=",
        "+",
        "*",
        "/",
        ")",
        "(",
        "[",
        "]",
        "{",
        "}",
        ",",
        "'",
        "′"
        "`",
        "~",
        "@",
        "&"
    ]
    
    for unwanted in unwanted:
        lyrics = lyrics.replace(unwanted, "")
    
    lyrics = lyrics.replace("â", "a")
    lyrics = lyrics.replace("ê", "e")
    lyrics = lyrics.replace("ô", "o")
    lyrics = lyrics.replace("û", "u")
    
    lyrics = lyrics.lower()

    lyrics = re.sub(r'\n', '.', lyrics)
    return lyrics

def process_data(lyrics_item):
    lyrics = lyrics_item[0]
    song_title = lyrics_item[1]
    artist_name = lyrics_item[2]

    # lyrics = "<|startofsong|><|startofline|>" + \
    #     lyrics + "<|endofline|><|endofsong|>"
    # replaced with "." instead
    # lyrics = re.sub(r'\n', '<|endofline|><|startofline|>', lyrics)
    
    lyrics = process_lyrics(lyrics)
    
    return {
        "lyrics": lyrics,
        "song_title": song_title,
        "artist_name": artist_name
    }
    

    
    
    
    
    