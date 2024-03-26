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
        try:
            self.conn = psycopg2.connect(
                host = self.host,
                dbname = self.dbname,
                port = self.port,
                user = self.user,
                password = self.password,
            )
            self.cur = self.conn.cursor()
            print("\n-# Connection & transaction with PostgreSQL : ON\n")
        except(Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL : ", error)
            sys.exit()

    def execute(self, query, placeHolderValues = None):
        self.check_connection()
        if placeHolderValues == None or None in placeHolderValues:
            self.cur.execute(query)
            print("-# " + query.as_string(self.conn) + ";\n")
        else:
            self.cur.execute(query, placeHolderValues)
            print("-# " + query.as_string(self.conn) % placeHolderValues + ";\n")

    # ============
    # CRUD methods        
    # ============

    def insert(self, **column_value):
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(self.table),
            sql.SQL(', ').join(map(sql.Identifier, column_value.keys())),
            sql.SQL(', ').join(sql.Placeholder() * len(column_value.values()))
        )

        record_to_insert = tuple(column_value.values())
        self.execute(insert_query, record_to_insert)
    
    def select(self, columns, searchKey = None, searchKeyValue = None):
        if searchKey == None:
            select_query = sql.SQL("SELECT {} FROM {}").format(
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.Identifier(self.table)
            )

            self.execute(select_query)
        else:
            select_query = sql.SQL("SELECT {} FROM {} WHERE {} = {}").format(
                sql.SQL(',').join(map(sql.Identifier, columns)),
                sql.Identifier(self.table),
                sql.Placeholder(),
                sql.Placeholder()
            )   

            self.execute(select_query, (searchKey, searchKeyValue))   
        try: 
            selected = self.cur.fetchall()
        except psycopg2.ProgrammingError as error:
            selected = '# ERROR:' + str(error)
        else:
            print(selected)
            print("rowcount : ", self.cur.rowcount)
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
            print("\n-# Connection & transaction with PostgreSQL : OFF\n")
            