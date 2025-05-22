from flask import Flask, render_template, redirect, url_for
from routes.prediction_routes import prediction_bp, load_model as load_prediction_model
from routes.dashboard_routes import dashboard_bp
from routes.fatality_risk_routes import fatality_risk_bp
from routes.summit_time_routes import summit_time_bp
from routes.peak_difficulty_routes import peak_difficulty_bp, load_model as load_peak_difficulty_model

app = Flask(__name__, static_folder='static')

# Register the blueprints
app.register_blueprint(prediction_bp, url_prefix='')
app.register_blueprint(dashboard_bp, url_prefix='')
app.register_blueprint(fatality_risk_bp)
app.register_blueprint(summit_time_bp)
app.register_blueprint(peak_difficulty_bp)

@app.route('/')
def home():
    return redirect(url_for('prediction.prediction_form'))

if __name__ == '__main__':
    # Load all models
    load_prediction_model()
    load_peak_difficulty_model()
    app.run(debug=True)
