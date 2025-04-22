# imports
import re
from flask import Flask, redirect, url_for, render_template, request

# settings (IP should be the intranet-public IP of the computer that executes the script)
max_rounds = 3
self_ip = "127.0.0.1" # add the intranet-public IP here, e.g., 192.168.1.1

# create flask app
app = Flask(__name__)


# method to reset the flask server
def clean():
    global round
    global left_score, left_name, left_spell_count, left_spell_css, left_spell_name, l_s_name, left_spell_class, l_s_class
    global right_score, right_name, right_spell_count, right_spell_css, right_spell_name, r_s_name, right_spell_class, r_s_class

    round = 1

    left_score = 0
    left_name = "left slot is empty"
    left_spell_count = 0
    left_spell_css = "neutral"
    left_spell_name = ""
    l_s_name = "no spell casted"
    left_spell_class = ""
    l_s_class = "no spell casted"

    right_score = 0
    right_name = "right slot is empty"
    right_spell_count = 0
    right_spell_css = "neutral"
    right_spell_name = ""
    r_s_name = "no spell casted"
    right_spell_class = ""
    r_s_class = "no spell casted"

# method that is called when the spells of both wizards are available and nee to be evaluated
def duel():
    global round, max_rounds
    global left_score, left_name, left_spell_count, left_spell_css, left_spell_name, l_s_name, left_spell_class, l_s_class
    global right_score, right_name, right_spell_count, right_spell_css, right_spell_name, r_s_name, right_spell_class, r_s_class

    # is the duel round not within the specified number of rounds?
    if round > max_rounds:
        print("[Duel] Maximum number of rounds played")
        return

    # increment the round number
    round += 1

    # make the currently casted spells visible 
    l_s_name = left_spell_name
    l_s_class = left_spell_class
    r_s_name = right_spell_name
    r_s_class = right_spell_class

    # filter the spell class number from the complete string
    i_left_class = int(re.sub("[^0-9]", "", left_spell_class))
    i_right_class = int(re.sub("[^0-9]", "", right_spell_class))

    # make sure that all spell classes are within the valid range 1-3
    if i_left_class > 3 or i_left_class < 1 or i_right_class > 3 or i_right_class < 1:
        print("[Duel] Invalid spell classes do not allow to compute winner")
        return

    # in case of draw
    if i_left_class == i_right_class:
        print("[Duel] Draw, all wizards receive a point")
        left_score += 1
        right_score += 1
        left_spell_css = "draw"
        right_spell_css = "draw"
        return
    # in case the left wizard wins
    elif i_left_class == 1 and i_right_class == 2 or i_left_class == 2 and i_right_class == 3 or i_left_class == 3 and i_right_class == 1:
        print("[Duel] Left wizard is able to beat right wizard")
        left_score += 1
        left_spell_css = "winner"
        right_spell_css = "loser"
        return
    # in case the right wizard wins
    else:
        print("[Duel] Right wizard is able to beat left wizard")
        right_score += 1
        left_spell_css = "loser"
        right_spell_css = "winner"
        return

# index / landing page
@app.route("/")
def home():
    return render_template("index.htm")

# arena page
@app.route("/arena", methods=["POST", "GET"])
def arena():
    global max_rounds

    # parse incoming data
    if request.method == 'POST':
        temp_round = request.form['rounds']
        reset = request.form['reset']
    else:
        temp_round = request.args.get('rounds')
        reset = request.args.get('reset')
    
    # check if maximum number of rounds has been updated
    if temp_round is not None and int(temp_round) > 0:
        max_rounds = int(temp_round)

    # check if arena should be reset
    if reset == 'on':
        clean()

    left_state = "normal"
    right_state = "normal"

    if round > max_rounds:
        if left_score > right_score:
            left_state = "victory"
        elif right_score > left_score:
            right_state = "victory"
        else:
            left_state = "victory"
            right_state = "victory"

    return render_template("arena.htm", 
        round=(round-1), max_rounds=max_rounds,
        left_state=left_state, right_state=right_state,
        left_score=left_score, right_score=right_score,
        left_name=left_name, right_name=right_name,
        left_spell_css=left_spell_css, right_spell_css=right_spell_css,
        left_spell_name=l_s_name, right_spell_name=r_s_name,
        left_spell_class=l_s_class, right_spell_class=r_s_class)

# cast spell page
@app.route("/castspell", methods=["POST", "GET"])
def castspell():
    global left_score, left_name, left_spell_count, left_spell_css, left_spell_name, l_s_name, left_spell_class, l_s_class
    global right_score, right_name, right_spell_count, right_spell_css, right_spell_name, r_s_name, right_spell_class, r_s_class
    
    # parameter: teamname (string), slot (1, 2), spellname (string), spellclass (1, 2, 3)
    if request.method == 'POST':
        teamname = request.form['teamname']
        slot = request.form['slot']
        spell_name = request.form['spellname']
        spell_class = request.form['spellclass']
    else:
        teamname = request.args.get('teamname')
        slot = request.args.get('slot')
        spell_name = request.args.get('spellname')
        spell_class = request.args.get('spellclass')

    if int(spell_class) == 1:
        spell_class = "Scissors (1)"
    elif int(spell_class) == 2:
        spell_class = "Paper (2)"
    elif int(spell_class) == 3:
        spell_class = "Rock (3)"
    else:
        spell_class = "Magic of Unknown Source"

    print("[Received Spell] Team: " + teamname + ", Slot: " + slot + ", Spell name: " + spell_name + ", Spell class: " + spell_class)

    if int(left_spell_count) >= max_rounds and int(right_spell_count) >= max_rounds:
        print("[Received Spell] Maximum number of spells has been casted already")
        return

    if int(slot) == 1:
        left_name = teamname
        left_spell_name = spell_name
        left_spell_class = spell_class
        l_s_name = "not unveiled yet"
        l_s_class = "not unveiled yet"
        if left_spell_count <= right_spell_count:
            left_spell_count += 1
            left_spell_css = "neutral"
            right_spell_css = "neutral"
            if left_spell_count == right_spell_count:
                duel()
            else:
                r_s_name = "no spell casted"
                r_s_class = "no spell casted"

    elif int(slot) == 2:
        right_name = teamname
        right_spell_name = spell_name
        right_spell_class = spell_class
        r_s_name = "not unveiled yet"
        r_s_class = "not unveiled yet"
        if right_spell_count <= left_spell_count:
            right_spell_count += 1
            right_spell_css = "neutral"
            left_spell_css = "neutral"
            if right_spell_count == left_spell_count:
                duel()
            else:
                l_s_name = "no spell casted"
                l_s_class = "no spell casted"

    return render_template("castspell.htm")

# main function
if __name__ == "__main__":
    clean()
    app.run(host=self_ip, debug=True)