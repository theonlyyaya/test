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
        conn = psycopg2.connect(
            host = self.host,
            database = self.dbname,
            user = self.user,
            password = self.password
        )

        
        
# https://github.com/Ahmed512w/Python-PostgreSQL-CRUD/blob/master/crud.py