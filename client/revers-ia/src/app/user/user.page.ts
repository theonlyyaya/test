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

  login() {
    // Rediriger vers la page de connexion
    this.router.navigate(['/login']);
  }

  signup() {
    // Rediriger vers la page d'inscription
    this.router.navigate(['/signup']);
  }
}
