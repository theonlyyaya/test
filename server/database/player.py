
from .databaseManager import DatabaseManager
from entities.player import Player


player = DatabaseManager(
    'postgres',
    'root',
    'localhost',
    '5432',
    'reversia',
    'Player',
    'id_player'
    )

players_table = player.select_all()

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

