class Model:
    id_model = 0
    model_name = ''
    white_win_rate = 0
    black_win_rate = 0
    file = ''
    id_opponent = 0

    def __init__(self, id_model, model_name, white_win_rate, black_win_rate, file, id_opponent):
        self.id_model = id_model
        self.model_name = model_name
        self.white_win_rate = white_win_rate
        self.black_win_rate = black_win_rate
        self.file = file
        self.id_opponent = id_opponent
