### Project Title: Heart Disease Classification and Analysis

---

### Dataset
The project utilizes the **Cleveland Heart Disease dataset** (`processed.cleveland.data`).
* **Target**: Binary classification (0: No disease, 1: Presence of disease).
* **Features**: 14 clinical attributes including age, sex, chest pain type (`cp`), cholesterol (`chol`), and maximum heart rate achieved (`thalach`).

---

### Project Structure
* **`/notebooks`**: Contains the main analysis file `i23_6004_DS_B_DMAssignment4 (1).ipynb`.
* **`/app`**: Contains `app.py` for the Streamlit web interface.
* **`/report`**: Place relevant project documentation and visual outputs (e.g., `correlation_heatmap.png`, `B1_confusion_matrix.png`) here.

---

### Download Steps
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```
2.  **Download the Dataset**: Ensure the file `processed.cleveland.data` is placed in the root directory.
3.  **Install Dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib shap xgboost tensorflow pyngrok streamlit
    ```

---

### How to Run

#### 1. Running the Notebook
* Open the notebook `i23_6004_DS_B_DMAssignment4 (1).ipynb` in Jupyter or Google Colab.
* Run the cells sequentially to perform:
    * **Data Preprocessing**: Handling missing values and scaling.
    * **Clustering**: KMeans and PCA analysis.
    * **Supervised Learning**: Training Random Forest, XGBoost, and Neural Networks.
    * **Model Evaluation**: Generating ROC curves and SHAP explanations.

#### 2. Running the App
The application is built with **Streamlit** for real-time prediction.
* Ensure the trained model files (e.g., `rf_model.pkl`, `xgb_model.pkl`) and the `scaler` are saved in the root directory.
* Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
* If running in a remote environment (like Colab), the notebook uses `pyngrok` to create a public URL for the app.
