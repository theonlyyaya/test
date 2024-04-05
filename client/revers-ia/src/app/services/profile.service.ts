import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ProfileService {
  constructor() {}

  getProfilePicture() {
    // Logique pour récupérer la photo de profil depuis le stockage
  }

  updateProfilePicture(newPicture: string) {
    // Logique pour mettre à jour la photo de profil dans le stockage
  }

  getUsername() {
    // Logique pour récupérer le nom d'utilisateur depuis le stockage
  }

  updateUsername(newUsername: string) {
    // Logique pour mettre à jour le nom d'utilisateur dans le stockage
  }

  getEmail() {
    // Logique pour récupérer l'email depuis le stockage
  }

  updateEmail(newEmail: string) {
    // Logique pour mettre à jour l'email dans le stockage
  }
}
