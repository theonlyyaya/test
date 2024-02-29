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

    def execute(self, query, placeholder_value = None):
        return 0

    # ============
    # CRUD methods
    # ============

    def insert(self, **column_value):
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})".format(
            sql.Identifier(self.table),
            sql.SQL(', ').join(map(sql.Identifier, column_value.keys())),
            sql.SQL(', ').join(sql.Placeholder() * len(column_value.values()))
        ))

        record_to_insert = tuple(column_value.values())
        self.execute(insert_query, record_to_insert)
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


