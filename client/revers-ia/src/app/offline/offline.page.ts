import { Component, OnInit } from '@angular/core';
import { ApiService } from '../services/api.service';
import { AlertController } from '@ionic/angular';

@Component({
  selector: 'app-offline',
  templateUrl: './offline.page.html',
  styleUrls: ['./offline.page.scss'],
})
export class OfflinePage implements OnInit {
  cells: string[][] = [];

  constructor(private apiService: ApiService, private alertController: AlertController) {}

  ngOnInit() {
    console.log('ngOnInit called');
    this.apiService.getBoard().subscribe(
      (board) => {
        console.log('Board state:', board);
        this.cells = board;
      },
      (error) => {
        console.error('Error fetching board:', error);
      }
    );
    const storedBoard = localStorage.getItem('reversi_board');
    if (storedBoard) {
        this.cells = JSON.parse(storedBoard);
    }
  }

  makeMove(row: number, col: number) {
    console.log(`Making move at ${row}, ${col}`);
    this.apiService.makeMove(row, col).subscribe((response) => {
      console.log('Move response:', response);
      const winner = response.headers.get('winner');
      if (winner) {
          this.displayWinnerMessage(winner);
      }
      this.apiService.getBoard().subscribe(
        (board) => {
          this.cells = board;
            // Save the updated board to local storage
            localStorage.setItem('reversi_board', JSON.stringify(board));
        },
        (error) => {
          console.error('Error fetching board:', error);
        }
      );
    });
    
  }

  displayWinnerMessage(winner: string) {
    // Utilise AlertController pour afficher le message du gagnant dans une popup
    this.alertController.create({
      header: 'Game Over',
      message: `The winner is ${winner === 'B' ? 'Player 1 (Black)' : 'Player 2 (White)'}`,
      buttons: ['OK']
    }).then(alert => alert.present());
    console.log('Winner:', winner);
  }

  calculateFloor(i: number): number {
    return Math.floor(i / 8);
  }

  calculateMod(i: number): number {
    return i % 8;
  }

  getImagePath(cell: string): string {
    if (cell === 'B') {
      return 'https://i.postimg.cc/t4QdpLGT/black-circle.png';
    }
    if (cell === 'W') {
      return 'https://i.postimg.cc/BvwB9tSW/white-circle.png';
    }
    return'';
  }
}
