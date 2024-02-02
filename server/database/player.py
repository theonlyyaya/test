from database.connect import db
from entities.player import Player
from dataset import Table

players_table = Table(db, 'players')  # Table(db['players'])

def addPlayer(player):
    found = getPlayer(player.id_player)
    if not found:
        players_table.insert(dict(player))
    return 0

def getPlayer(id_player):
    player = players_table.find_one(id=id_player)
    return player
