# IMPORTANT: Please make all your changes to this file
import pandas as pd
from random import randrange

spellname1 = "Spell #1"
spellname2 = "Spell #2"
spellname3 = "Spell #3"

def process_spell(pandas_df: pd.DataFrame):
    # give a number for predicted spell (1 to 3; with 1 wins over 2, 2 wins over 3, and 3 wins over 1)
    predicted_spell = randrange(3) + 1

    print("Incoming Dataframe:")
    print(pandas_df.head(5))

    # TODO use your preprocessing from your Jupyter Notebook here and load your model
    #
    # look here for how to save and load a model: 
    # https://www.geeksforgeeks.org/saving-a-machine-learning-model/

    return (predicted_spell, get_spellname(predicted_spell))

def get_spellname(id):
    if id == 1:
        return spellname1
    if id == 2:
        return spellname2
    if id == 3:
        return spellname3
    return "Unknown Spell"