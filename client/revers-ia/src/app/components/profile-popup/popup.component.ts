import { Component, OnInit } from '@angular/core';
import { PopoverController } from '@ionic/angular';
import { Router } from '@angular/router';

@Component({
  selector: 'app-profile-popup',
  templateUrl: './profile-popup.component.html',
  styleUrls: ['./profile-popup.component.scss'],
})
export class ProfilePopupComponent implements OnInit {

  constructor(private popoverController: PopoverController, private router: Router) { }

  ngOnInit() {}

  login() {
    // Rediriger vers la page de connexion
    this.router.navigate(['/login']);
    this.popoverController.dismiss();
  }

  signup() {
    // Rediriger vers la page d'inscription
    this.router.navigate(['/signup']);
    this.popoverController.dismiss();
  }
}
