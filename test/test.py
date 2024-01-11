from flask import Flask, jsonify, request
from flask_cors import CORS
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label

app = Flask(__name__)
CORS(app)

class ReversiGrid(GridLayout):
    def __init__(self, **kwargs):
        super(ReversiGrid, self).__init__(**kwargs)
        self.cols = 8
        self.rows = 8
        self.board = [[' ' for _ in range(8)] for _ in range(8)]
        self.current_player = 'B'
        self.create_board()
        self.place_initial_pieces()

    def create_board(self):
        for row in range(8):
            for col in range(8):
                cell = Button(background_color=(1, 1.8, 1, 1.8))  # Light green background
                cell.bind(on_press=self.make_move)
                
                if self.board[row][col] == 'B':
                    cell.background_normal = 'black_circle.png'
                elif self.board[row][col] == 'W':
                    cell.background_normal = 'white_circle.png'
                
                self.add_widget(cell)

    def place_initial_pieces(self):
        self.board[3][3] = 'W'
        self.board[4][4] = 'W'
        self.board[3][4] = 'B'
        self.board[4][3] = 'B'
        for i, child in enumerate(reversed(self.children)):
            row, col = self.get_coords(child)
            if (row, col) in [(3, 3), (4, 4)]:
                child.background_normal = 'white_circle.png'
            elif (row, col) in [(3, 4), (4, 3)]:
                child.background_normal = 'black_circle.png'

    def get_coords(self, instance):
        index = self.children.index(instance)
        row = index // self.cols
        col = index % self.cols
        return row, col
    
    def is_valid_move(self, row, col):
        if self.board[row][col] != ' ':
            return False  # La case est déjà occupée, le mouvement est donc invalide

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_move = False

        for dx, dy in directions:
            r, c = row + dx, col + dy
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != ' ' and self.board[r][c] != self.current_player:
                to_flip.append((r, c))
                r += dx
                c += dy
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player and to_flip:
                valid_move = True
                break

        return valid_move

    def update_board(self):
        for i, child in enumerate(reversed(self.children)):
            row, col = self.get_coords(child)
            if self.board[row][col] == 'B':
                child.background_normal = 'black_circle.png'  
            elif self.board[row][col] == 'W':
                child.background_normal = 'white_circle.png' 
    
    def make_move(self, instance):
        row, col = self.get_coords(instance)
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.flip_pieces(row, col)
            self.update_board()
            self.current_player = 'W' if self.current_player == 'B' else 'B'
            # Vérifie s'il n'y a plus de mouvements possibles
            if not any(self.is_valid_move(row, col) for row in range(8) for col in range(8)):
                black_count, white_count = self.count_pieces()
                winner = "Black" if black_count > white_count else "White" if white_count > black_count else "Draw"
                winner_text = f"The winner is {winner}!"
                popup_content = Label(text=winner_text)
                popup = Popup(title="Game Over", content=popup_content, size_hint=(None, None), size=(400, 200))
                popup.open()
            
    def flip_pieces(self, row, col):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            r, c = row + dx, col + dy
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != ' ' and self.board[r][c] != self.current_player:
                to_flip.append((r, c))
                r += dx
                c += dy
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player:
                for r, c in to_flip:
                    self.board[r][c] = self.current_player
    
    def count_pieces(self):
        black_count = sum(row.count('B') for row in self.board)
        white_count = sum(row.count('W') for row in self.board)
        return black_count, white_count   

# Crée une instance du jeu Reversi
reversi_game = ReversiGrid() 

class ReversiApp(App):
    def build(self):
        return ReversiGrid()
    
    
@app.route('/api/get_board', methods=['GET'])
def get_board():
    return jsonify(reversi_game.board)


@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.get_json()
    row = data['row']
    col = data['col']
    
    # Your existing logic for making a move goes here

    return jsonify({"success": True})

if __name__ == "__main__":
    reversi_game = ReversiGrid()
    ReversiApp().run()