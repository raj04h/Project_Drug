from flask import Flask, request, jsonify
import pickle
import pandas as pd


app = Flask(__name__)
# 1) Load your trained pipeline
with open('ml_Drug.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

@app.route('/')
def home():
    return "IC50 Prediction Service"

@app.route('/predict', methods=['POST'])
def predict():
    # Expect either form-data or JSON
    drug_name    = request.form.get('Drug name')    or request.json.get('Drug name')
    drug_target  = request.form.get('Drug target')  or request.json.get('Drug target')
    target_path  = request.form.get('Target Pathway') or request.json.get('Target Pathway')
    feature_name = request.form.get('Feature Name') or request.json.get('Feature Name')
    
    # Numeric inputs (cast to float)
    n_pos = float(request.form.get('n_feature_pos') or request.json.get('n_feature_pos'))
    n_neg = float(request.form.get('n_feature_neg') or request.json.get('n_feature_neg'))
   
    # 2) Build DataFrame with exactly the same columns & order as training
    input_df = pd.DataFrame([{
        'n_feature_pos': n_pos,
        'n_feature_neg': n_neg,
        'Drug target': drug_target,
        'Target Pathway': target_path,
        'Drug name': drug_name,
        'Feature Name': feature_name,
    }])

    # 3) Predict
    pred = model_pipeline.predict(input_df)[0]

    return jsonify({'ic50_effect_size': float(pred)})

if __name__ == '__main__':
    app.run(debug=True)
