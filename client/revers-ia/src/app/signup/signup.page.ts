import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CountryService } from '../country.service';

@Component({
  selector: 'app-signup',
  templateUrl: './signup.page.html',
  styleUrls: ['./signup.page.scss'],
})
export class SignupPage implements OnInit {
  countries: any[] = [];
  username: string = '';
  email: string = '';
  password: string = '';
  confirmPassword: string = '';
  country: string = '';

  constructor(private router: Router, private countryService: CountryService) { }

  ngOnInit() {
    this.countryService.getCountries().subscribe((data) => {
      this.countries = data.map((country) => ({
        name: country.name.common, // Adaptez cette ligne si la structure de données de l'API change
        code: country.cca2
      })).sort((a, b) => a.name.localeCompare(b.name));
    });
  }

  signUp() {
    // Logique d'inscription
    console.log('Username:', this.username);
    console.log('Email:', this.email);
    console.log('Password:', this.password);
    console.log('Confirm Password:', this.confirmPassword);
    console.log('Country:', this.country);

    // Redirection après inscription
    this.router.navigate(['/login']); // Assurez-vous que le chemin est correct
  }
}
