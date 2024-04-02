import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-user',
  templateUrl: './user.page.html',
  styleUrls: ['./user.page.scss'],
})
export class UserPage implements OnInit {

  constructor(private router: Router) { }

  ngOnInit() {
  }

  goToLogin() {
    // Rediriger vers la page de connexion
    this.router.navigate(['/login']);
  }

  goToSignup() {
    // Rediriger vers la page d'inscription
    this.router.navigate(['/signup']);
  }
}
