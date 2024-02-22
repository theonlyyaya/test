import dataset

#db = dataset.connect('postgresql://postgres:root@localhost:5432/reversia')

# CREATE TABLE `reversia`.`users` (`userId` BIGINT NOT NULL AUTO_INCREMENT , `username` VARCHAR(50) NOT NULL , `email` VARCHAR(255) NOT NULL , `password` VARCHAR(50) NOT NULL , PRIMARY KEY (`userId`), UNIQUE (`email`)) ENGINE = MyISAM; 
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="reversia",
    user="postgres",
    password="root"
)

cur = conn.cursor()

# Création de la table Opponent (une entité participant au jeu)
cur.execute("""
    CREATE TABLE Opponent(
    id_opponent INT,
    PRIMARY KEY(id_opponent)
    )
""")

# Création de la table Game (une partie de Reversi)
cur.execute("""
    CREATE TABLE Game(
    id_game INT,
    started_at DATETIME,
    ended_at DATETIME,
    data VARCHAR(120),
    PRIMARY KEY(id_game)
    )     
""")

# Création de la table DLmodel
# Un modèle est un opposant
cur.execute("""
    CREATE TABLE DLmodel(
    id_model INT,
    model_name VARCHAR(50) NOT NULL,
    white_win_rate DECIMAL(15,2) NOT NULL,
    black_win_rate DECIMAL(15,2) NOT NULL,
    file VARCHAR(255),
    id_opponent INT NOT NULL,
    PRIMARY KEY(id_model),
    UNIQUE(id_opponent),
    UNIQUE(model_name),
    FOREIGN KEY(id_opponent) REFERENCES Opponent(id_opponent)
    )
""")

# Création de la table Player
# Un joueur est un opposant
cur.execute("""
    CREATE TABLE Player(
    id_player INT,
    pseudo VARCHAR(50) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password VARCHAR(50) NOT NULL,
    created_at DATE NOT NULL,
    id_opponent INT NOT NULL,
    PRIMARY KEY(id_player),
    UNIQUE(id_opponent),
    UNIQUE(email),
    FOREIGN KEY(id_opponent) REFERENCES Opponent(id_opponent)
    )
""")

cur.close()
conn.commit()
conn.close()