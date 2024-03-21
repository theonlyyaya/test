import { Component, OnInit, ChangeDetectorRef, ChangeDetectionStrategy } from '@angular/core';
import { ApiService } from '../services/api.service';
import { AlertController } from '@ionic/angular';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-offline',
  templateUrl: './offline.page.html',
  styleUrls: ['./offline.page.scss'],
  changeDetection: ChangeDetectionStrategy.Default,
})
export class OfflinePage implements OnInit {
  cells: number[][] = [];
  private moveMadeSubscription: Subscription = new Subscription();
  player1Score: number = 0;
  player2Score: number = 0;
  activePlayer: number = -1; // Par défaut, le joueur 1 est actif

  constructor(
    private apiService: ApiService,
    private alertController: AlertController  ) {}

  ngOnInit() {
    // Initialize the board and subscribe to move events
    this.initBoard();
    this.subscribeToMoves();
  }

  ngOnDestroy() {
    // Unsubscribe to prevent memory leaks
    if (this.moveMadeSubscription) {
      this.moveMadeSubscription.unsubscribe();
    }
  }

  initBoard() {
    this.apiService.getBoard().subscribe(
      (board) => {
        this.cells = board;
        this.updateScores();
      },
      (error) => {
        console.error('Error fetching board:', error);
      }
    );
  }

  subscribeToMoves() {
    this.moveMadeSubscription = this.apiService.onMoveMade().subscribe(() => {
      this.apiService.getBoard().subscribe(
        (board) => {
          if (JSON.stringify(this.cells) !== JSON.stringify(board)){
            this.toggleTurn(); // Change the turn after each move
          }
          this.cells = board;
          this.updateScores();
          localStorage.setItem('reversi_board', JSON.stringify(board));
        },
        (error) => {
          console.error('Error fetching board:', error);
        }
      );
    });
  }

  makeMove(row: number, col: number) {
    this.apiService.makeMove(row, col).subscribe(
      (response) => {
        const winner = response.winner;
        if (winner) {
          this.displayWinnerMessage(winner);
        } else {
          // Continue with the game logic if there is no winner
          this.initBoard();
        }
      },
      (error) => {
        // Handle errors here
        console.error('Error making move:', error);
      }
    );
  }

  updateScores() {
    this.player1Score = this.countPieces(-1);
    this.player2Score = this.countPieces(1);
  }

  countPieces(player: number): number {
    return this.cells.reduce((count, row) => count + row.filter((cell) => cell === player).length, 0);
  }

  toggleTurn() {
    // Changer le joueur actif
    this.activePlayer *= -1;
  }

  displayWinnerMessage(winner: string) {
    this.alertController.create({
      header: 'Game Over',
      message: `The winner is ${winner === 'Black' ? 'Player 1 (Black)' : 'Player 2 (White)'}`,
      buttons: [
        {
          text: 'Play Again',
          handler: () => {
            location.reload();
          },
        },
      ],
    }).then((alert) => alert.present());
    console.log(winner);
  }

  getImagePath(cell: number): string {
    if (cell === -1) {
      return 'https://i.postimg.cc/t4QdpLGT/black-circle.png';
    }
    if (cell === 1) {
      return 'https://i.postimg.cc/HWRrXXx8/white-circle.png';
    }
    return '';
  }
}
