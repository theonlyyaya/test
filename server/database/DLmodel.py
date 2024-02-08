from dataset import Table
from database.connect import db
from entities.model import Model

models_table = Table(db, 'dlmodels')

def addModel():
    return 0