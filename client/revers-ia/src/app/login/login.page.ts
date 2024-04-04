// login.page.ts
import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.page.html',
  styleUrls: ['./login.page.scss'],
})
export class LoginPage {
  email: string = '';
  password: string = '';

  constructor(private router: Router) { }

  login() {
    // Ici, vous pouvez ajouter votre logique d'authentification
    console.log('Email:', this.email);
    console.log('Password:', this.password);

    // Supposons que l'authentification est réussie, vous pouvez rediriger l'utilisateur
    // Remplacez '/dashboard' par le chemin de votre page de dashboard ou page principale
    this.router.navigate(['/dashboard']);
  }

  navigateToSignup() {
    // Navigue vers la page d'inscription
    this.router.navigate(['/signup']);
  }
}
