import psycopg2

class PostgresQuery:
    def __init__(self):
        self.connection = psycopg2.connect(database="postgres", user="postgres", password="docker", host="localhost", port=5432)
        self.cursor = self.connection.cursor()

    def create_table(self):
        query = """
                CREATE TABLE IF NOT EXISTS dataset (
                    id SERIAL PRIMARY KEY,
                    song_id VARCHAR(255),
                    artist_id TEXT,
                    song TEXT,
                    artists TEXT,
                    explicit BOOLEAN,
                    genres TEXT,
                    lyrics TEXT
                );

                CREATE INDEX IF NOT EXISTS lyrics_idx ON dataset USING gist(to_tsvector('english', lyrics));
                """
        self.cursor.execute(query)
        self.connection.commit()

    def query_search(self, query_text, top_k):
        query_text = query_text.replace(" ", " | ")
        query = """
                SELECT song_id, artists, lyrics, ts_rank(to_tsvector('english', lyrics), to_tsquery('english', '%s')) AS similitud
                FROM dataset
                WHERE to_tsvector('english', lyrics) @@ to_tsquery('english', '%s')  
                ORDER BY similitud DESC 
                LIMIT %i;
                """
        self.cursor.execute(query, (query_text, query_text, top_k))
        return self.cursor.fetchall()