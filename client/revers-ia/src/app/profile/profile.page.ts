import { Component } from '@angular/core';
import { ProfileService } from '../services/profile.service';

@Component({
  selector: 'app-profile',
  templateUrl: './profile.page.html',
  styleUrls: ['./profile.page.scss'],
})
export class ProfilePage {
  username: string | undefined;
  email: string | undefined;
  profilePicture: string | undefined;

  constructor(private profileService: ProfileService) { }

  ngOnInit() {
    this.getUsername();
    this.getEmail();
    this.getProfilePicture();
  }

  getUsername() {
    const username = this.profileService.getUsername();
    if (username !== undefined) {
      this.username = username;
    }
  }

  getEmail() {
    const email = this.profileService.getEmail();
    if (email !== undefined) {
      this.email = email;
    }
  }

  getProfilePicture() {
    const picture = this.profileService.getProfilePicture();
    if (picture !== undefined) {
      this.profilePicture = picture;
    }
  }

  updateUsername(newUsername: string) {
    this.profileService.updateUsername(newUsername);
    this.getUsername(); // Met à jour la valeur affichée après la mise à jour
  }

  updateEmail(newEmail: string) {
    this.profileService.updateEmail(newEmail);
    this.getEmail(); // Met à jour la valeur affichée après la mise à jour
  }

  updateProfilePicture(newPicture: string) {
    this.profileService.updateProfilePicture(newPicture);
    this.getProfilePicture(); // Met à jour la valeur affichée après la mise à jour
  }

  onProfilePictureChange(event: any) {
    const file = event.target.files[0];
    // Ajoutez ici la logique pour mettre à jour la photo de profil avec le fichier sélectionné
  }
  
  saveChanges() {
    // Vérifiez si this.username est défini
    if (this.username !== undefined) {
      // Si c'est le cas, appelez la méthode d'update du service ProfileService pour mettre à jour le nom d'utilisateur
      this.updateUsername(this.username);
    } else {
      // Sinon, affichez un message d'erreur ou gérez la situation selon vos besoins
      console.error("Le nom d'utilisateur est indéfini.");
    }
    if (this.email !== undefined) {
      // Si c'est le cas, appelez la méthode d'update du service ProfileService pour mettre à jour le nom d'utilisateur
      this.updateUsername(this.email);
    } else {
      // Sinon, affichez un message d'erreur ou gérez la situation selon vos besoins
      console.error("email est indéfini.");
    }
  
  }
}
