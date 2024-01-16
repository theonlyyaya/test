import { Component, OnInit, ChangeDetectorRef, ChangeDetectionStrategy } from '@angular/core';
import { ApiService } from '../services/api.service';
import { AlertController } from '@ionic/angular';
import { Subscription } from 'rxjs';
import { exec } from 'child_process';


@Component({
  selector: 'app-offline',
  templateUrl: './offline.page.html',
  styleUrls: ['./offline.page.scss'],
  changeDetection: ChangeDetectionStrategy.Default,
})
export class OfflinePage implements OnInit {
  cells: string[][] = [];
  private moveMadeSubscription: Subscription = new Subscription();

  constructor(
    private apiService: ApiService,
    private alertController: AlertController,
    private changeDetectorRef: ChangeDetectorRef
  ) {}

  ngOnInit() {
    this.apiService.getBoard().subscribe(
      (board) => {
        this.cells = board;
      },
      (error) => {
        console.error('Error fetching board:', error);
      }
    );

    this.moveMadeSubscription = this.apiService.onMoveMade().subscribe(() => {
      this.apiService.getBoard().subscribe(
        (board) => {
          this.cells = board;
          this.changeDetectorRef.markForCheck(); // This marks the component for check
          localStorage.setItem('reversi_board', JSON.stringify(board));
        },
        (error) => {
          console.error('Error fetching board:', error);
        }
      );
    });
  }

  ngOnDestroy() {
    // Unsubscribe to prevent memory leaks
    if (this.moveMadeSubscription) {
      this.moveMadeSubscription.unsubscribe();
    }
  }

  makeMove(row: number, col: number) {
    this.apiService.makeMove(row, col).subscribe(
      (response) => {
        const winner = response.winner;
        if (winner) {
          this.displayWinnerMessage(winner);
        } else {
          // Continue with the game logic if there is no winner
          this.apiService.getBoard().subscribe(
            (board) => {
              this.cells = board;
              localStorage.setItem('reversi_board', JSON.stringify(board));
            },
            (error) => {
              console.error('Error fetching board:', error);
            }
          );
        }
      },
      (error) => {
        // Handle errors here
        console.error('Error making move:', error);
      }
    );
  }



  displayWinnerMessage(winner: string) {
    this.alertController.create({
      header: 'Game Over',
      message: `The winner is ${winner === 'Black' ? 'Player 1 (Black)' : 'Player 2 (White)'}`,
      buttons: [{
        text: 'Play Again',
        handler: () => {
          location.reload();;
        }
      }],
    }).then((alert) => alert.present());
    console.log(winner);
  }

  getImagePath(cell: string): string {
    if (cell === 'B') {
      return 'https://i.postimg.cc/t4QdpLGT/black-circle.png';
    }
    if (cell === 'W') {
      return 'https://i.postimg.cc/HWRrXXx8/white-circle.png';
    }
    return '';
  }
}
