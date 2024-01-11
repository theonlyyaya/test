// offline.page.ts

import { Component, OnInit } from '@angular/core';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-offline',
  templateUrl: './offline.page.html',
  styleUrls: ['./offline.page.scss'],
})
export class OfflinePage implements OnInit {
  cells: string[][] = [];

  constructor(private apiService: ApiService) {}

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
      this.apiService.getBoard().subscribe(
        (board) => {
          this.cells = board;
          const winner = response.winner;
            if (winner) {
                this.displayWinnerMessage(winner);
            }

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
    // Display your winner message in the UI
    console.log('Winner:', winner);
}

  calculateFloor(i: number): number {
    return Math.floor(i / 8);
  }

  calculateMod(i: number): number {
    return i % 8;
  }
}
