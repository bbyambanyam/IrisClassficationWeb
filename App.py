from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired

import classifier

app = Flask(__name__)

# Flask-WTF requires an encryption key
app.config['SECRET_KEY'] = 'IrisClassificationWeb'

# Flask-Bootstrap requires this line
Bootstrap(app)

class NameForm(FlaskForm):
    model = SelectField(u'Model', choices=[('dtc', 'DecisionTreeClassifier'), ('gnb', 'GaussianNB'), ('knc', 'KNeighborsClassifier')])
    sepalLength = StringField('Sepal length:', validators=[DataRequired()])
    sepalWidth = StringField('Sepal width:', validators=[DataRequired()])
    petalLength = StringField('Petal length:', validators=[DataRequired()])
    petalWidth = StringField('Petal width:', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = NameForm()
    result = ""
    if form.validate_on_submit():
        sl = form.sepalLength.data
        sw = form.sepalWidth.data
        pl = form.petalLength.data
        pw = form.petalWidth.data
        mdl = form.model.data
        result = classifier.classify(sl, sw, pl, pw, mdl)
    return render_template('index.html', form=form, result=result)

if __name__ == '__main__':
    app.run(debug=True)

