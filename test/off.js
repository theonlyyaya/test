class ReversiGrid {
    constructor() {
        this.cols = 8;
        this.rows = 8;
        this.board = Array.from({ length: 8 }, () => Array(8).fill(' '));
        this.currentPlayer = 'B';
        this.initializeBoard();
    }

    initializeBoard() {
        const grid = document.getElementById('grid');

        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                const cell = document.createElement('button');
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.classList.add('cell');
                cell.addEventListener('click', () => this.handleClick(row, col));
                grid.appendChild(cell);
            }
        }

        this.placeInitialPieces();
        this.updateBoard();
    }

    placeInitialPieces() {
        this.board[3][3] = 'W';
        this.board[4][4] = 'W';
        this.board[3][4] = 'B';
        this.board[4][3] = 'B';

        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                if (this.board[row][col] === 'W') {
                    cell.classList.add('white');
                } else if (this.board[row][col] === 'B') {
                    cell.classList.add('black');
                }
            }
        }
    }

    handleClick(row, col) {
        if (this.isValidMove(row, col)) {
            this.board[row][col] = this.currentPlayer;
            this.flipPieces(row, col);
            this.updateBoard();
            this.currentPlayer = this.currentPlayer === 'B' ? 'W' : 'B';
            this.checkEndGame();
        }
    }

    isValidMove(row, col) {
        if (this.board[row][col] !== ' ') {
            return false;
        }

        const directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ];

        let validMove = false;

        directions.forEach(([dx, dy]) => {
            let r = row + dx;
            let c = col + dy;
            let toFlip = [];

            while (r >= 0 && r < this.rows && c >= 0 && c < this.cols && this.board[r][c] !== ' ' && this.board[r][c] !== this.currentPlayer) {
                toFlip.push([r, c]);
                r += dx;
                c += dy;
            }

            if (
                r >= 0 && r < this.rows &&
                c >= 0 && c < this.cols &&
                this.board[r][c] === this.currentPlayer &&
                toFlip.length > 0
            ) {
                toFlip.forEach(([r, c]) => {
                    this.board[r][c] = this.currentPlayer;
                });
                validMove = true;
            }
        });

        return validMove;
    }

    updateBoard() {
        const grid = document.getElementById('grid');

        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                const cell = grid.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                cell.classList.remove('white', 'black');
                if (this.board[row][col] === 'W') {
                    cell.classList.add('white');
                } else if (this.board[row][col] === 'B') {
                    cell.classList.add('black');
                }
            }
        }
    }

    flipPieces(row, col) {
        const directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ];
    
        directions.forEach(([dx, dy]) => {
            let r = row + dx;
            let c = col + dy;
            let toFlip = [];
    
            while (
                r >= 0 && r < this.rows &&
                c >= 0 && c < this.cols &&
                this.board[r][c] !== ' ' &&
                this.board[r][c] !== this.currentPlayer
            ) {
                toFlip.push([r, c]);
                r += dx;
                c += dy;
            }
    
            if (
                r >= 0 && r < this.rows &&
                c >= 0 && c < this.cols &&
                this.board[r][c] === this.currentPlayer &&
                toFlip.length > 0
            ) {
                toFlip.forEach(([r, c]) => {
                    this.board[r][c] = this.currentPlayer;
                });
            }
        });
    }
    

    checkEndGame() {
        let hasValidMoves = false;
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (this.isValidMove(row, col)) {
                    hasValidMoves = true;
                    break;
                }
            }
            if (hasValidMoves) {
                break;
            }
        }
    
        if (!hasValidMoves) {
            const blackCount = this.board.reduce((acc, row) => acc + row.filter(cell => cell === 'B').length, 0);
            const whiteCount = this.board.reduce((acc, row) => acc + row.filter(cell => cell === 'W').length, 0);
            
            if (blackCount > whiteCount) {
                console.log('Black wins!');
            } else if (whiteCount > blackCount) {
                console.log('White wins!');
            } else {
                console.log('It\'s a draw!');
            }
        }
    }
    
}

// Création d'une instance du jeu et initialisation
const game = new ReversiGrid();
