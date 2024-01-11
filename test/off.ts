const board = document.getElementById("board");

const player1 = "black";
const player2 = "white";

let currentPlayer = player1;

const boardState: string[][] = new Array(8)
  .fill("")
  .map(() => new Array(8).fill(""));

function createBoard() {
  if (board) {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        cell.dataset.row = `${i}`;
        cell.dataset.col = `${j}`;
        cell.addEventListener("click", handleCellClick);
        board.appendChild(cell);
        boardState[i][j] = "";
      }
    }
    // Placer les pions de départ
    placeInitialPieces();
  } else {
    console.error("Impossible de trouver l'élément avec l'ID 'board'");
  }
}

function placeInitialPieces() {
  const initialPieces = [
    [3, 3],
    [4, 4],
    [3, 4],
    [4, 3],
  ];

  initialPieces.forEach(([row, col]) => {
    const cell = document.querySelector(
      `[data-row="${row}"][data-col="${col}"]`
    ) as HTMLElement;
    const piece = document.createElement("div");
    piece.classList.add("piece", currentPlayer);
    cell.appendChild(piece);
    boardState[row][col] = currentPlayer;
  });
}

function handleCellClick(event: Event) {
  const clickedCell = event.target as HTMLElement;
  const row = parseInt(clickedCell.dataset.row || "0");
  const col = parseInt(clickedCell.dataset.col || "0");

  if (isValidMove(row, col)) {
    // Mettre à jour le tableau et le visuel
    updateBoard(row, col);
    // Changer de joueur
    currentPlayer = currentPlayer === player1 ? player2 : player1;
  }
}

function isValidMove(row: number, col: number): boolean {
  // Vérifier si le coup est légal
  // Retourner true si c'est le cas, sinon false
  return true; // À implémenter avec les règles du Reversi
}

function updateBoard(row: number, col: number) {
  const cell = document.querySelector(
    `[data-row="${row}"][data-col="${col}"]`
  ) as HTMLElement;
  const piece = document.createElement("div");
  piece.classList.add("piece", currentPlayer);
  cell.appendChild(piece);
  boardState[row][col] = currentPlayer;
}

createBoard();
