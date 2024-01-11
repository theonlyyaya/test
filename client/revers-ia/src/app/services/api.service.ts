// api.service.ts

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private apiUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) {}

  getBoard(): Observable<any> {
    return this.http.get(`${this.apiUrl}/get_board`).pipe(
      catchError((error) => {
        console.error('Error fetching board:', error);
        throw error;
      })
    );
  }

  makeMove(row: number, col: number): Observable<any> {
    return this.http.post(`${this.apiUrl}/make_move`, { row, col });
  }
}
