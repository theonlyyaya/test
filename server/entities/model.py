class Model:
    id_model = 0
    model_name = ''
    win_rate = 0
    file = ''
    id_opponent = 0

    def __init__(self, id_model, model_name, win_rate, file, id_opponent):
        self.id_model = id_model
        self.model_name = model_name
        self.win_rate = win_rate
        self.file = file
        self.id_opponent = id_opponent
