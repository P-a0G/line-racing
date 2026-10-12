
Récapitulatif des nouveautés

Les parties peuvent désormais se jouer à 3 joueurs.
Consultez l'énoncé mis à jour pour plus de détails.
 	Objectif
Dans ce jeu vous êtes un programme aux commandes du bolide mythique de tron : le lumicycle. Vous affrontez sans cesse d'autres programmes sur la Grille dans des combats homériques de lumicycles.

Le lumicycle se déplace en ligne droite et ne peut tourner qu'à 90° laissant un ruban de lumière solide dans son sillage. Chaque lumicycle et son ruban associé ont une couleur différente sur la Grille.
Si un lumicycle s'arrête, heurte un ruban de lumière ou sort de la Grille, il sera instantanément mis hors-jeu.

Le dernier cycle en jeu gagne la partie. Votre objectif est de devenir le meilleur programme : une fois envoyés dans l'arène, les programmes vont s'affronter dans des matchs regroupant de deux à quatre lumicycles. Plus vous gagnez de combats, meilleur sera votre classement.
 	Règles
Chaque combat regroupe 2 ou 3 joueurs. Chaque joueur joue à tour de rôle. Quand votre tour vient, les événements suivants s'enchaînent :
Les informations de position des joueurs sur la Grille sont envoyées sur l'entrée standard (stdin). Votre IA devrait donc être en attente, en lecture sur l'entrée standard, au début d'un tour de jeu.
Une fois les informations lues, votre IA doit calculer le prochain mouvement du lumicycle et l'écrire sur la sortie standard (stdout). Pour un tour de jeu, la sortie doit être une unique ligne indiquant la nouvelle direction du lumicycle: UP ou DOWN ou LEFT ou RIGHT.
Votre lumicycle se déplace dans la direction indiquée par votre IA.
A partir de ce moment, votre IA devrait se mettre en attente des informations du tour suivant et ainsi de suite... Dans l'intervalle, l'IA des autres joueurs recevra les informations de jeu de la même façon.
Si votre IA ne calcule pas son mouvement assez vite, ou si vous indiquez un mouvement non valide, ou si votre lumicycle se heurte à un obstacle après s'être déplacé, alors votre programme perd le combat.

Si une autre IA perd avant la vôtre, son ruban de lumière disparaît et le jeu continue jusqu'à ce qu'il n'y ait plus qu'un seul joueur sur la Grille.

La Grille est composée de cases ( 30 cases par 20 cases). Chaque joueur commence à une position aléatoire sur la Grille.
 
Conditions de victoire
Être le dernier joueur restant
 	Entrées du jeu
Entrée pour un tour de jeu
Ligne 1 : deux entiers N et P. Où N est le nombre total de joueurs et P désigne votre numéro de joueur pour la partie en cours.

Les N lignes suivantes : une ligne par joueur. La première ligne est pour le joueur 0, la ligne suivante est pour le joueur 1, etc. Chaque ligne liste quatre entiers X0, Y0, X1 et Y1. (X0, Y0) sont les coordonnées de la position initiale du lumicycle (queue du ruban) et (X1, Y1) sont les coordonnées de la position courante du lumicycle du joueur (tête du ruban). Une fois qu'un joueur perd, ses coordonnées X0 Y0 X1 Y1 sont toutes égales à -1 (le lumicycle et le ruban de lumière du joueur disparaissent de la Grille).

Sortie pour un tour de jeu
Une ligne unique composée du mot UP, DOWN, LEFT ou RIGHT
Contraintes
2 ≤ N ≤ 3
0 ≤ P < N
0 ≤ X0, X1 < 30
0 ≤ Y0, Y1 < 20
Votre IA doit répondre en moins de 100ms à chaque tour.
Exemple
Une bataille à deux joueurs. Votre joueur commence à la position (9,5). L'autre joueur commence à la position (10,7). Vous jouez en premier.
Entrée pour le tour 1 (votre tour)
2 0         (N P)
9 5 9 5	    (X0 Y0 X1 Y1)
10 7 10 7   (X0 Y0 X1 Y1)
Sortie pour le tour 1
RIGHT
Déplacement de la position (9,5) vers (10,5)
Entrée pour le tour 2 (ce que l'autre joueur lit)
2 1         (N P)
9 5 10 5    (X0 Y0 X1 Y1)
10 7 10 7   (X0 Y0 X1 Y1)
Sortie pour le tour 2
UP
Déplacement de la position (10,7) vers (10,6)
Entrée pour le tour 3 (votre tour)
2 0         (N P)
9 5 10 5    (X0 Y0 X1 Y1)
10 7 10 6   (X0 Y0 X1 Y1)
Sortie pour le tour 3
UP
Déplacement de la position (10,5) vers (10,4)
Entrée pour le tour 4 (ce que l'autre joueur lit)
2 1         (N P)
9 5 10 4    (X0 Y0 X1 Y1)
10 7 10 6   (X0 Y0 X1 Y1)
Sortie pour le tour 4
UP
Déplacement de la position (10,6) vers (10,5)

L'autre joueur perd car son lumicycle entre en collision avec votre ruban de lumière. Vous gagnez le combat !