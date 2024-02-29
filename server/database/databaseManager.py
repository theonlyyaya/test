import psycopg2
import psycopg2.sql as sql

class DatabaseManager:
    def __init__(self, user, password, host, port, dbname, table, primarykey):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname
        self.table = table
        self.primarykey = primarykey

    def connect(self):
        self.conn = psycopg2.connect(
            host = self.host,
            database = self.dbname,
            user = self.user,
            password = self.password
        )
        self.cur = self.conn.cursor()

    # ============
    # CRUD methods
    # ============

    def insert(self):
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})")
        return 0
    
    def select(self):
        return 0

    def update(self):
        return 0
    
    def delete(self):
        return 0
    

    def close(self, commit = False):
        self.cur.close()
        self.conn.close()


