from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

app = Flask(__name__)

# === INIT: Load saat server pertama kali dijalankan ===
# Load CSV sebagai DataFrame
supermarket_df = pd.read_csv('data/supermarket_encoded.csv')

# Load encoder
user_encoder = joblib.load('models/user_encoder.pkl')
product_encoder = joblib.load('models/product_encoder.pkl')

# Load model Keras
model = load_model('models/supermarket_recommender.keras')

# Buat list semua produk (encoded)
num_items = len(product_encoder.classes_)
all_products = np.arange(num_items)

# === ROUTE: Predict Rekomendasi Produk ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_user_id = data.get('id_user')
    count_items = int(data.get('count_items', 10))

    try:
        # Transform user ID ke encoded
        target_user = user_encoder.transform([raw_user_id])[0]

        # Ambil produk yang sudah dibeli user
        user_encoded_col = user_encoder.transform(supermarket_df['id_user'])
        supermarket_df['user_id_enc'] = user_encoded_col
        product_encoded_col = product_encoder.transform(supermarket_df['id_product'])
        supermarket_df['product_id_enc'] = product_encoded_col

        rated_products = supermarket_df[supermarket_df['user_id_enc'] == target_user]['product_id_enc'].tolist()

        # Produk yang belum dibeli
        all_products = np.arange(len(product_encoder.classes_))
        unrated_products = np.setdiff1d(all_products, rated_products)

        if len(unrated_products) == 0:
            return jsonify({
                "user_id": raw_user_id,
                "message": "Semua produk telah dibeli oleh user ini.",
                "recommendations": []
            })

        # Buat array user
        user_array = np.full(len(unrated_products), target_user)

        # Prediksi rating
        pred_ratings = model.predict([user_array, unrated_products]).flatten()

        # DataFrame hasil prediksi
        results = pd.DataFrame({
            'product_id_enc': unrated_products,
            'predicted_quantity': pred_ratings,
        })
        results['id_product'] = product_encoder.inverse_transform(results['product_id_enc'])

        # Gabungkan nama dan kategori produk
        results = results.merge(
            supermarket_df[['id_product', 'name', 'category']].drop_duplicates(),
            on='id_product',
            how='left'
        )

        # Ambil rekomendasi teratas
        top_recommendations = results.sort_values('predicted_quantity', ascending=False).head(count_items)

        return jsonify({
            'user_id': raw_user_id,
            'recommendations': top_recommendations[['id_product', 'name', 'category', 'predicted_quantity']].to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True)
