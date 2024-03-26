
from datetime import datetime
import operator
from sympy import true
from databaseManager import DatabaseManager
import hashlib

DATABASE_USER = 'postgres'
DATABASE_PASSWORD = 'root'
DATABASE_HOST = 'localhost'
DATABASE_PORT = '5432'
DATABASE_NAME = 'reversia'
TABLE_NAME = 'players'
TABLE_PK = 'id_player'

player = DatabaseManager(
    DATABASE_USER,
    DATABASE_PASSWORD,
    DATABASE_HOST,
    DATABASE_PORT,
    DATABASE_NAME,
    TABLE_NAME,
    TABLE_PK
)

player.connect()

#players_table = player.select_all()

#def find_one(email):
#    return player.select_all("email", email)

def signUp(email, password, confirmedPassword, pseudo):
    if confirmedPassword == password:
        encodedPassword = password.encode()
        hash1 = hashlib.md5(encodedPassword).hexdigest()
        player.insert(
            email=email,
            password=hash1,
            pseudo=pseudo,
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')     
        )
    else:
        return "passwords are not the same!"
    
def logIn(email, password):
    auth_password = password.encode()
    auth_hash = hashlib.md5(auth_password).hexdigest()
    #stored_email = find_one(email=email)
    stored_password = player.select(columns=['password'], searchKey="email", searchKeyValue="'"+email+"'")
    
    print("stored password : ", stored_password)
    print("\nauth_password : ", auth_hash)
    if stored_password == auth_hash:
        return True
    else:
        return "wrong password!"

#print(signUp("salut@some.com", "patate", "patate", "thib"))
print(logIn('salut@some.com', "patate"))

player.close(commit=True)
player.close()