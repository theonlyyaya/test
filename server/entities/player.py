class Player:
    id_player = 0
    pseudo = ''
    email = ''
    password = ''
    id_opponent = 0
    created_at = ''

    def __init__(self, id_player, pseudo, email, password, id_opponent):
        self.id_player = id_player
        self.pseudo = pseudo
        self.email = email
        self.password = password
        self.id_opponent = id_opponent
        self.created_at = '' # A modifier

    
