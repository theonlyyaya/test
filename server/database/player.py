from database.connect import db
from entities.player import Player
from dataset import Table

players_table = Table(db, 'players')

def addPlayer(player):
    found = getPlayer(player.id_player)
    if not found:
        players_table.insert(dict(player))
    return 0

def getPlayer(id_player):
    player = players_table.find_one(id=id_player)
    if(player):
        player = Player(**(player))
    return player

def logIn(email, password):
    player = players_table.find_one(email=email)
#    if(player.password == password):
#        return True
    return False 



def signIn(email, password):
    """
    Parameters
    ----------
    email,
    password
    """
    player = players_table.find_one(email=email)
    if(player):
        return False
    return True