import psycopg2
import csv

class PostgresQuery:
    def __init__(self):
        self.connection = psycopg2.connect(database="postgres", user="postgres", password="docker", host="localhost", port=5432)
        self.cursor = self.connection.cursor()

    def table_exists(self):
        query = """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = 'songs'
                );
                """
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]

    def create_table(self):
        if self.table_exists():
            print('La tabla de PostgreSQL ya existe. Omitiendo creación e inserción.')
            return
        
        query = """
                CREATE TABLE songs(
                    id SERIAL PRIMARY KEY,
                    track_id VARCHAR(255),
                    track_name TEXT,
                    track_artist TEXT,
                    lyrics TEXT,
                    track_popularity VARCHAR(255),
                    track_album_id VARCHAR(255),
                    track_album_name TEXT,
                    track_album_release_date VARCHAR(255),
                    playlist_name TEXT,
                    playlist_id VARCHAR(255),
                    playlist_genre TEXT,
                    playlist_subgenre TEXT,
                    danceability NUMERIC,
                    energy NUMERIC,
                    key INT,
                    loudness NUMERIC,
                    mode INT,
                    speechiness NUMERIC,
                    acousticness NUMERIC,
                    instrumentalness NUMERIC,
                    liveness NUMERIC,
                    valence NUMERIC,
                    tempo NUMERIC,
                    duration_ms INT,
                    language TEXT
                );

                CREATE INDEX IF NOT EXISTS lyrics_idx ON songs USING gist(to_tsvector('english', lyrics));
                """
        self.cursor.execute(query)
        self.connection.commit()

        self.insert_data_from_csv()
    
    def insert_data_from_csv(self):
        print('Insertando datos a tabla de PostgreSQL...')
        with open('Proyecto 2/utils/spotify_songs.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.cursor.execute("""
                    INSERT INTO songs (
                        track_id, track_name, track_artist, lyrics, track_popularity,
                        track_album_id, track_album_name, track_album_release_date,
                        playlist_name, playlist_id, playlist_genre, playlist_subgenre,
                        danceability, energy, key, loudness, mode, speechiness,
                        acousticness, instrumentalness, liveness, valence, tempo,
                        duration_ms, language
                    ) VALUES (
                        %(track_id)s, %(track_name)s, %(track_artist)s, %(lyrics)s,
                        %(track_popularity)s, %(track_album_id)s, %(track_album_name)s,
                        %(track_album_release_date)s, %(playlist_name)s, %(playlist_id)s,
                        %(playlist_genre)s, %(playlist_subgenre)s, %(danceability)s,
                        %(energy)s, %(key)s, %(loudness)s, %(mode)s, %(speechiness)s,
                        %(acousticness)s, %(instrumentalness)s, %(liveness)s, %(valence)s,
                        %(tempo)s, %(duration_ms)s, %(language)s
                    )
                """, row)

        self.connection.commit()

    def query_search(self, query_text, top_k):
        query_text = query_text.replace(" ", " | ")
        query = """
                SELECT track_id, track_name, track_artist, playlist_genre, lyrics, 
                       ts_rank(to_tsvector('english', lyrics), to_tsquery('english', %s)) AS score
                FROM songs
                WHERE to_tsvector('english', lyrics) @@ to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s;
                """
        self.cursor.execute(query, (query_text, query_text, top_k))
        results = self.cursor.fetchall()

        formatted_res = []
        for song_data in results:
            data = {
                "song_id": song_data[0],
                "song": song_data[1],
                "artist": song_data[2],
                "genre": song_data[3],
                "score": round(song_data[5], 4),
                "lyrics": song_data[4]
            }
            formatted_res.append(data)
        
        return formatted_res
    
if __name__ == "__main__":
    postgres = PostgresQuery()
    postgres.create_table()
    postgres.insert_data_from_csv()