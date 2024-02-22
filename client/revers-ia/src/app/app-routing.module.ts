import { NgModule } from '@angular/core';
import { PreloadAllModules, RouterModule, Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    loadChildren: () =>
      import('./tabs/tabs.module').then((m) => m.TabsPageModule),
  },  {
    path: 'offline',
    loadChildren: () => import('./offline/offline.module').then( m => m.OfflinePageModule)
  },
  {
    path: 'player-vs-ai',
    loadChildren: () => import('./player-vs-ai/player-vs-ai.module').then( m => m.PlayerVsAiPageModule)
  },
  {
    path: 'ai-vs-ai',
    loadChildren: () => import('./ai-vs-ai/ai-vs-ai.module').then( m => m.AiVsAiPageModule)
  },
  {
    path: 'how-to-play',
    loadChildren: () => import('./how-to-play/how-to-play.module').then( m => m.HowToPlayPageModule)
  },

];
@NgModule({
  imports: [
    RouterModule.forRoot(routes, { preloadingStrategy: PreloadAllModules }),
  ],
  exports: [RouterModule],
})
export class AppRoutingModule {}
