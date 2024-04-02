import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-signup',
  templateUrl: './signup.page.html',
  styleUrls: ['./signup.page.scss'],
})
export class SignupPage implements OnInit {

  username: string | undefined;
  email: string | undefined;
  password: string | undefined;

  constructor(private router: Router) { }

  ngOnInit() {
  }

  signUp() {
    // Vous pouvez ajouter ici la logique pour envoyer les données d'inscription à votre backend
    console.log('Username:', this.username);
    console.log('Email:', this.email);
    console.log('Password:', this.password);

    // Une fois l'inscription réussie, vous pouvez rediriger l'utilisateur vers une autre page
    this.router.navigate(['/login']); // Remplacez '/home' par le chemin de la page où vous souhaitez rediriger l'utilisateur après l'inscription
  }

}
