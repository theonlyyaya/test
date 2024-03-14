
from sympy import true
from .databaseManager import DatabaseManager
from entities.player import Player
import hashlib

DATABASE_USER = 'postgres'
DATABASE_PASSWORD = 'root'
DATABASE_HOST = 'localhost'
DATABASE_PORT = '5432'
DATABASE_NAME = 'reversia'
TABLE_NAME = 'Player'
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

#players_table = player.select_all()

def addPlayer(player):
    found = getPlayer(player.id_player)
    if not found:
        player.insert(dict(player))
    return 0

def getPlayer(id_player):
    player = players_table.find_one(id=id_player)
    if(player):
        player = Player(**(player))
    return player

def logIn(email, password):
    player = players_table.find_one(email=email)
    if(player):
        player = Player(**(player))
        if(player.password == password):
            return True
    return False 

def logIn(email, password):
    player = players_table.find_one(email=email)

def find_one(string):
    return string


def signIn(email, password):
    """
    Parameters
    ----------
    email,
    password
    """
    player = players_table.find_one(email=email)
    if(player):
        player = Player(**(player))
        if(password == player.password):
            player.password = ''
    return player

def find_one(email):
    return player.select(["email"], email)

def signUp(email, password, confirmedPassword, pseudo):
    if confirmedPassword == password:
        encodedPassword = password.encode()
        hash1 = hashlib.md5(encodedPassword).hexdigest()
        player.insert(email=email, password=hash1, pseudo=pseudo)
    else:
        return "passwords are not the same!"
    
def logIn(email, password):
    auth_password = password.encode()
    auth_hash = hashlib.md5(auth_password).hexdigest()
    stored_email = find_one(email=email)
    stored_password = player.select('password', stored_email)
    if stored_password == auth_hash and email == stored_email:
        return True
    else:
        return "wrong password!"