-- SQLite


SELECT count(*) FROM lyrics;

-- get artist count
SELECT count(*) FROM (SELECT DISTINCT artist_name FROM lyrics);

-- get unsued query count
SELECT count(*) FROM queries WHERE is_used = FALSE;