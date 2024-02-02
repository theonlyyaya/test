class Game:
    id_game = 0
    started_at = ''
    ended_at = ''
    id_opponent1 = 0
    id_opponent2 = 0

    def __init__(self, id_game, started_at, ended_at, id_opponent1, id_opponent2):
        self.id_game = id_game
        self.started_at = started_at # /!\ A modifier... (date)
        self.ended_at = ended_at     # Idem
        self.id_opponent1 = id_opponent1
        self.id_opponent2 = id_opponent2