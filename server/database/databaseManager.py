import psycopg2

class DatabaseManager:
    def __init__(self, user, password, host, port, dbname, primarykey):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname
        self.primarykey = primarykey

    def connect(self):
        self.conn = psycopg2.connect(
            host = self.host,
            database = self.dbname,
            user = self.user,
            password = self.password
        )

    def insert(self):
        return 0

    def close(self):
        self.conn.close()

