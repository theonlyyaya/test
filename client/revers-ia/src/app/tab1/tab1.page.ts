import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { PopoverController } from '@ionic/angular';
import { ProfilePopupComponent } from '../components/profile-popup/popup.component';

@Component({
  selector: 'app-tab1',
  templateUrl: 'tab1.page.html',
  styleUrls: ['tab1.page.scss']
})

export class Tab1Page {

  constructor(private router: Router, public popoverController: PopoverController) {}

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

  async openProfilePopup() {
    const popover = await this.popoverController.create({
      component: ProfilePopupComponent,
      translucent: true,
      cssClass: 'profile-popup-class' // Définissez une classe CSS pour personnaliser l'apparence du popup
    });
    return await popover.present();
  }

}
