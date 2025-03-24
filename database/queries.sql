-- SQLite


SELECT count(*) from songs;

-- get artist count
SELECT count(*) FROM (SELECT DISTINCT artist_name from songs);

-- get unsued query count
SELECT count(*) FROM queries WHERE is_used = FALSE;