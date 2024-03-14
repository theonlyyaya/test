import psycopg2, sys
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

    def execute(self, query, placeHolderValues = None):
        self.check_connection()
        if placeHolderValues == None or None in placeHolderValues:
            self.cur.execute(query)
        else:
            self.cur.execute(query, placeHolderValues)

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
    
    def select(self, columns, pk = None):
        if pk == None:
            select_query = sql.SQL("SELECT {} FROM {}").format(
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.Identifier(self.table)
            )

            self.execute(select_query)
        else:
            select_query = sql.SQL("SELECT {} FROM {} WHERE {} = {}").format(
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.Identifier(self.table),
                sql.Identifier(self.primarykey),
                sql.Placeholder()
            )   

            self.execute(select_query, pk)
        
        selected = self.cur.fetchall()
        return selected

    def select_all(self):
        select_querry = sql.SQL("SELECT * FROM {}").format(
            sql.Identifier(self.table)
        )

        self.execute(select_querry)
        selected = self.cur.fetchall()
        return selected

    def update_column(self, column, column_value, pk):
        update_query = sql.SQL("UPDATE {} SET {} = {} WHERE {} = {}").format(
            sql.Identifier(self.table),
            sql.Identifier(column),
            sql.Placeholder(),
            sql.Identifier(self.primarykey),
            sql.Placeholder()
        )

        self.execute(update_query, (column_value, pk))
    
    def update_multiple_columns(self, columns, columns_value, pk):
        update_query = sql.SQL("UPDATE {} SET ({}) = ({}) WHERE {} = {}").format(
            sql.Identifier(self.table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns_value)),
            sql.Identifier(self.primarykey),
            sql.Placeholder()
        )

        placeHolderValues = list(columns_value)
        placeHolderValues.append(pk)
        placeHolderValues = tuple(placeHolderValues)
        self.execute(update_query, placeHolderValues)

    def delete(self, pk):
        delete_query = sql.SQL("DELETE FROM {} WHERE {} = {}").format(
            sql.Identifier(self.table),
            sql.Identifier(self.primarykey),
            sql.Placeholder()
        )

        self.execute(delete_query, pk)
    
    # ============

    def check_connection(self):
        try:
            self.conn
        except AttributeError:
            sys.exit()


    # Commit changes
    def commit(self):
        self.check_connection()
        self.conn.commit()

    # Close connection
    def close(self, commit = False):
        self.check_connection()
        if commit:
            self.commit()
        else:
            self.cur.close()
            self.conn.close()
            