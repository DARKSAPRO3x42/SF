# app.py
from flask import Flask, request, jsonify, render_template
from sklearn.neural_network import MLPRegressor
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. SOFT COMPUTING: ANN MODEL SETUP
# ==========================================
# Features: [processing_time, memory_req, urgency_level (1-10)]
# Target: Priority score (higher score = schedule earlier)
X_train = np.array([
    [10, 2, 8], 
    [5, 1, 3], 
    [20, 8, 9], 
    [2, 1, 1], 
    [15, 4, 5]
])
y_train = np.array([85, 30, 95, 10, 60])

# Initialize and train the Artificial Neural Network
# Hidden layers: One layer of 10 neurons, one layer of 5 neurons
ann_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)

# ==========================================
# 2. WEB SERVER ROUTES
# ==========================================
@app.route('/')
def home():
    # Serves the frontend UI
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    jobs = data.get('jobs', [])
    
    results = []
    # Feed each job into the Neural Network to get a priority prediction
    for job in jobs:
        features = np.array([[job['processing_time'], job['memory'], job['urgency']]])
        priority_prediction = ann_model.predict(features)[0]
        
        results.append({
            'job_id': job['id'],
            'priority_score': round(float(priority_prediction), 2)
        })
    
    # Soft Computing Optimization: Sort jobs by highest priority score
    optimized_schedule = sorted(results, key=lambda x: x['priority_score'], reverse=True)
    
    return jsonify({'optimized_schedule': optimized_schedule})

if __name__ == '__main__':
    app.run(debug=True)