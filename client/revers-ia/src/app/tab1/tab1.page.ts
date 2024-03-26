import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-tab1',
  templateUrl: 'tab1.page.html',
  styleUrls: ['tab1.page.scss']
})

export class Tab1Page {

  constructor(private router: Router) {}

  redirectToUserPage() {
    this.router.navigate(['/user']);// Rediriger vers la page de connexion
  }

  goToOfflinePage() {
    this.router.navigate(['/offline']); // Remplace '/offline' par le chemin de ta page "offline"
  }

  GoToChoosePlayerDisc() {
    this.router.navigate(['/choose-disc']);
  }

  goToAIVsAIPage() {
    this.router.navigate(['/ai-vs-ai']); // Remplace '/ai-vs-ai' par le chemin de ta page "ai-vs-ai"
  }

  goToHowToPlay() {
    this.router.navigate(['/how-to-play']); // Remplace '/how-to-play' par le chemin de ta page "how-to-play"
  }
}
