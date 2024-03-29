from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import torch

BOARD_SIZE = 8
app = Flask(__name__)
CORS(app)


def initialize_board():
    # Create an initial Othello board with zeros
    board_stat_init = np.zeros((BOARD_SIZE, BOARD_SIZE))

    # Place initial Othello pieces on the board
    board_stat_init[3, 3] = 1  # White piece
    board_stat_init[4, 4] = 1  # White piece
    board_stat_init[3, 4] = -1  # Black piece
    board_stat_init[4, 3] = -1  # Black piece

    return board_stat_init


def is_valid_coord(row, col, board_size=8):
    ''' Method: is_valid_coord
        Parameters: self, row (integer), col (integer)
        Returns: boolean (True if row and col is valid, False otherwise)
        Does: Checks whether the given coordinate (row, col) is valid.
              A valid coordinate must be in the range of the board.
    '''
    if 0 <= row < board_size and 0 <= col < board_size:
        return True
    return False


def has_tile_to_flip(move, direction, board_stat, NgBlackPsWhith):
    ''' Method: has_tile_to_flip
        Parameters: move (tuple), direction (tuple)
        Returns: boolean 
                 (True if there is any tile to flip, False otherwise)
        Does: Checks whether the player has any adversary's tile to flip
              with the move they make.

              About input: move is the (row, col) coordinate of where the 
              player makes a move; direction is the direction in which the 
              adversary's tile is to be flipped (direction is any tuple 
              defined in MOVE_DIRS).
    '''
    i = 1
    if is_valid_coord(move[0], move[1]):
        while True:
            row = move[0] + direction[0] * i
            col = move[1] + direction[1] * i
            if not is_valid_coord(row, col) or \
                    board_stat[row][col] == 0:
                return False
            elif board_stat[row][col] == NgBlackPsWhith:
                break
            else:
                i += 1
    return i > 1


def is_legal_move(move, board_stat, NgBlackPsWhith):
    ''' Method: is_legal_move
        Parameters: move (tuple)
        Returns: boolean (True if move is legal, False otherwise)
        Does: Checks whether the player's move is legal.

              About input: move is a tuple of coordinates (row, col).
    '''
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
                 (0, -1), (0, +1),
                 (+1, -1), (+1, 0), (+1, +1)]

    if move != () and is_valid_coord(move[0], move[1]) \
            and board_stat[move[0]][move[1]] == 0:
        for direction in MOVE_DIRS:
            if has_tile_to_flip(move, direction, board_stat, NgBlackPsWhith):
                return True
    return False


def get_legal_moves(board_stat, NgBlackPsWhith):
    ''' Method: get_legal_moves
        Parameters: self
        Returns: a list of legal moves that can be made
        Does: Finds all the legal moves the current player can make.
              Every move is a tuple of coordinates (row, col).
    '''
    moves = []
    for row in range(len(board_stat)):
        for col in range(len(board_stat)):
            move = (row, col)
            if is_legal_move(move, board_stat, NgBlackPsWhith):
                moves.append(move)
    return moves


def input_seq_generator(board_stats_seq, length_seq):

    board_stat_init = initialize_board()

    if len(board_stats_seq) >= length_seq:
        input_seq = board_stats_seq[-length_seq:]
    else:
        input_seq = [board_stat_init]
        # Padding starting board state before first index of sequence
        for i in range(length_seq - len(board_stats_seq) - 1):
            input_seq.append(board_stat_init)
        # adding the initial of game as the end of sequence sample
        for i in range(len(board_stats_seq)):
            input_seq.append(board_stats_seq[i])

    return input_seq


def find_best_move(move1_prob, legal_moves):
    """
    Finds the best move based on the provided move probabilities and legal moves.

    Parameters:
    - move1_prob (numpy.ndarray): 2D array representing the probabilities of moves.
    - legal_moves (list): List of legal moves.

    Returns:
    - tuple: The best move coordinates (row, column).
    """

    # Initialize the best move with the first legal move
    best_move = legal_moves[0]

    # Initialize the maximum score with the probability of the first legal move
    max_score = move1_prob[legal_moves[0][0], legal_moves[0][1]]

    # Iterate through all legal moves to find the one with the maximum probability
    for i in range(len(legal_moves)):
        # Update the best move if the current move has a higher probability
        if move1_prob[legal_moves[i][0], legal_moves[i][1]] > max_score:
            max_score = move1_prob[legal_moves[i][0], legal_moves[i][1]]
            best_move = legal_moves[i]
    return best_move


class ReversiGrid():
    def __init__(self):
        self.cols = 8
        self.rows = 8
        self.initialize()

    def initialize(self):
        self.board = initialize_board()
        self.current_player = -1  # -1 is black, 1 is white

    def is_valid_move(self, row, col):
        if self.board[row][col] != 0:
            return False  # La case est déjà occupée, le mouvement est donc invalide

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_move = False

        for dx, dy in directions:
            r, c = row + dx, col + dy
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != 0 and self.board[r][c] != self.current_player:
                to_flip.append((r, c))
                r += dx
                c += dy
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player and to_flip:
                valid_move = True
                break

        return valid_move

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.flip_pieces(row, col)
            self.current_player = -1 if self.current_player == 1 else 1
            black_count, white_count = self.count_pieces()

            # Check if there are no more valid moves
            if not any(self.is_valid_move(row, col) for row in range(8) for col in range(8)):
                black_count, white_count = self.count_pieces()
                winner = "Black" if black_count > white_count else "White" if white_count > black_count else "Draw"

                # Return the result with winner information
                return {"success": True, "winner": winner}

        # If there are still valid moves, return success without winner information
        return {"success": True}

    def make_one_move(self, playerDisc, player):  # player = difficulty (type of AI)
        # player: model description
        # board_stat: current 8x8 board status
        # turn: 1 or -1 - black or white turn
        # if current move is for player, skip
        if ((self.current_player == -1 and playerDisc == 'Black') or (self.current_player == 1 and playerDisc == 'White')):
            return -1, -1
        device = torch.device("cpu")

        conf = {}
        if (player == 'Easy'):
            conf['player'] = ''
        elif (player == 'Medium'):
            conf['player'] = ''
        elif (player == 'Hard'):
            conf['player'] = 'models\\Hard.pt'

        model = torch.load(conf['player'], map_location=torch.device('cpu'))
        model.eval()
        input_seq_boards = input_seq_generator(self.board, model.len_inpout_seq)

        # if black is the current player the board should be multiplay by -1
        if (self.current_player == -1):
            model_input = np.array([input_seq_boards]) * -1
        else:
            model_input = np.array([input_seq_boards])
        move1_prob = model(torch.tensor(model_input).float().to(device))
        move1_prob = move1_prob.cpu().detach().numpy().reshape(8, 8)
        legal_moves = get_legal_moves(self.board, self.current_player)
        if len(legal_moves) > 0:
            best_move = find_best_move(move1_prob, legal_moves)
            if (self.current_player == -1):
                print(f"Black: {best_move} < from possible move {legal_moves}")
            else:
                print(f"White: {best_move} < from possible move {legal_moves}")
            return best_move

    def flip_pieces(self, row, col):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            r, c = row + dx, col + dy
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != 0 and self.board[r][c] != self.current_player:
                to_flip.append((r, c))
                r += dx
                c += dy
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player:
                for r, c in to_flip:
                    self.board[r][c] = self.current_player

    def count_pieces(self):
        black_count = sum(row.count(-1) for row in self.board)
        white_count = sum(row.count(1) for row in self.board)
        return black_count, white_count


# Create an instance of the ReversiGrid game
reversi_game = ReversiGrid()


@app.route('/get_board', methods=['GET'])
def get_board():
    return jsonify(reversi_game.board)


@app.route('/get_possible_moves', methods=['GET'])
def get_possible_moves():
    return jsonify(get_legal_moves(reversi_game.board, reversi_game.current_player))


@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.get_json()
    row = data['row']
    col = data['col']

    result = reversi_game.make_move(row, col)
    # Extract the winner information from the result
    winner = result.get("winner")

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust the origin based on your requirements
    if winner:
        # Add the winner information to the response
        response.headers.add('winner', winner)  # type: ignore

    return response


@app.route('/make_one_move', methods=['POST'])
def make_one_move():
    data = request.get_json()
    difficulty = data['difficulty']
    playerDisc = data['playerDisc']
    row, col = reversi_game.make_one_move(playerDisc, difficulty)  # type: ignore
    if (row == -1 or col == -1):
        row = data['row']
        col = data['col']
    result = reversi_game.make_move(row, col)
    # Extract the winner information from the result
    winner = result.get("winner")

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust the origin based on your requirements
    if winner:
        # Add the winner information to the response
        response.headers.add('winner', winner)  # type: ignore

    return response


@app.route('/reload', methods=['GET'])
def reload_board():
    reversi_game.initialize()
    return jsonify(reversi_game.board)
