import os
import time
import datetime
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, make_response
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(
    IMAGES_FOLDER=os.path.join(app.root_path, 'static'),
    SECRET_KEY='96341257'
))
app.config["CACHE_TYPE"] = "null"
app.config.from_envvar('masters_chance_of_admit_SETTINGS', silent=True)


@app.route('/')
def show_analytics():
    """
    Show the analytics page.
    """
    return render_template('show_analytics.html')

LIMITS = {
    'gre': (0, 340),
    'cgpa': (1, 10),
    'toefl': (1, 120),
    'univ': (1, 5),
    'sop': (1, 5),
    'lor': (1, 5)
}

def verify_and_make_inputs(form, keys) -> np.ndarray:
    result = []

    for key in keys:
        user_input = form[key]
        value = int(user_input)

        MIN, MAX = LIMITS[key]

        if not MIN <= value <= MAX:
            raise ValueError(f'Value of {key} is not within limits.')

        result.append(value)

    result.append(1 if 'research' in form.keys() else 0)

    return np.array(result)


@app.route('/calculate_chance', methods=['GET', 'POST'])
def calculate_chance():
    """
    Calculate user's chances based on inputs
    """
    error = None
    try:
        if request.method == 'POST':
            if('toefl_req' in request.form.keys()):
                with open(os.path.join(app.root_path,'models', 'model_xtra_trees.pkl'), 'rb') as f:
                    model = pickle.load(f)

                features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
                print(1 if 'research' in request.form.keys() else 0)

                form_keys = ['gre', 'toefl', 'univ', 'sop', 'lor', 'cgpa']
            else:
                with open(os.path.join(app.root_path,'models', 'model_xtra_trees_no_toefl.pkl'), 'rb') as f:
                    model = pickle.load(f)

                features = ['GRE Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

                form_keys = ['gre', 'univ', 'sop', 'lor', 'cgpa']

            inputs = verify_and_make_inputs(request.form, form_keys)
            pred = model.predict(inputs)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(inputs)
            image_loc = os.path.join(app.config['IMAGES_FOLDER'], 'shap.png')
            if(os.path.isfile(image_loc)):
                os.remove(image_loc)
            shap.force_plot(explainer.expected_value, shap_values, pd.DataFrame(inputs, columns=features), matplotlib=True, show=False)
            plt.savefig(image_loc)
            return render_template('chance.html', chance = str(pred[0]*100)[:5], image_loc = image_loc, rand=random.random())       #random number to prevent cached images from being displayed
    except Exception as e:
        error = "There was an error processing your request. Please make sure you entered the right values or try again."

        error = '\n\n'.join([error, str(e)])

    return render_template('calculate_chance.html', error = error)
