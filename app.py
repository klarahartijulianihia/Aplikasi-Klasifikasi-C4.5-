import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from io import StringIO
import base64

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Klasifikasi C4.5",
    page_icon="🌳",
    layout="wide"
)

# Judul aplikasi
st.title("🌳 Aplikasi Klasifikasi dengan Algoritma C4.5")
st.markdown("---")

class C45Classifier:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.class_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def preprocess_data(self, data, target_column):
        """Preprocess data untuk klasifikasi"""
        try:
            # Pisahkan features dan target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle categorical features dengan one-hot encoding
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            
            # Simpan feature names
            self.feature_names = X.columns
            self.class_names = y.unique()
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error dalam preprocessing data: {str(e)}")
            return False
    
    def train(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """Train model Decision Tree (C4.5)"""
        try:
            self.model = DecisionTreeClassifier(
                criterion='entropy',  # C4.5 menggunakan entropy
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            self.model.fit(self.X_train, self.y_train)
            return True
            
        except Exception as e:
            st.error(f"Error dalam training model: {str(e)}")
            return False
    
    def predict(self, X_new):
        """Prediksi data baru"""
        if self.model is None:
            st.error("Model belum ditraining!")
            return None
        
        try:
            # Preprocess data baru sama seperti training data
            categorical_columns = X_new.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                X_new = pd.get_dummies(X_new, columns=categorical_columns, drop_first=True)
            
            # Pastikan feature sama dengan training
            X_new = X_new.reindex(columns=self.feature_names, fill_value=0)
            
            predictions = self.model.predict(X_new)
            return predictions
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")
            return None
    
    def evaluate(self):
        """Evaluasi model"""
        if self.model is None:
            st.error("Model belum ditraining!")
            return None
        
        try:
            # Predict pada training dan test set
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            
            # Calculate accuracy
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_test_pred': y_test_pred,
                'classification_report': classification_report(self.y_test, y_test_pred, output_dict=True)
            }
            
        except Exception as e:
            st.error(f"Error dalam evaluasi: {str(e)}")
            return None

def main():
    classifier = C45Classifier()
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
    app_mode = st.sidebar.selectbox(
        "Pilih Mode",
        ["Upload Data", "Preprocessing", "Training Model", "Evaluasi", "Prediksi"]
    )
    
    # Session state untuk menyimpan data
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Mode 1: Upload Data
    if app_mode == "Upload Data":
        st.header("📁 Upload Data")
        
        # Opsi upload file
        uploaded_file = st.file_uploader(
            "Upload file CSV atau Excel",
            type=['csv', 'xlsx'],
            help="Upload dataset dalam format CSV atau Excel"
        )
        
        if uploaded_file is not None:
            try:
                # Baca file berdasarkan tipe
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                # Tampilkan preview data
                st.subheader("Preview Data")
                st.write(f"Shape: {data.shape}")
                st.dataframe(data.head())
                
                # Tampilkan info data
                st.subheader("Info Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Kolom:**")
                    for col in data.columns:
                        st.write(f"- {col}")
                
                with col2:
                    st.write("**Tipe Data:**")
                    st.write(data.dtypes)
                
                # Simpan data ke session state
                st.session_state.data = data
                st.success("✅ Data berhasil diupload!")
                
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
    
    # Mode 2: Preprocessing
    elif app_mode == "Preprocessing":
        st.header("🔧 Preprocessing Data")
        
        if st.session_state.data is None:
            st.warning("⚠️ Silakan upload data terlebih dahulu di tab 'Upload Data'")
            return
        
        data = st.session_state.data
        
        # Pilih target column
        st.subheader("Pilih Target Variable")
        target_column = st.selectbox(
            "Pilih kolom target untuk klasifikasi:",
            options=data.columns
        )
        
        # Tampilkan distribusi target
        st.write("**Distribusi Target:**")
        target_counts = data[target_column].value_counts()
        st.bar_chart(target_counts)
        
        if st.button("Proses Preprocessing"):
            with st.spinner("Memproses data..."):
                success = classifier.preprocess_data(data, target_column)
                
                if success:
                    st.session_state.target_column = target_column
                    st.session_state.trained = False
                    
                    st.success("✅ Preprocessing berhasil!")
                    
                    # Tampilkan info preprocessing
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Training Set:**")
                        st.write(f"- Jumlah sample: {len(classifier.X_train)}")
                        st.write(f"- Jumlah feature: {len(classifier.feature_names)}")
                    
                    with col2:
                        st.write("**Test Set:**")
                        st.write(f"- Jumlah sample: {len(classifier.X_test)}")
                        st.write(f"- Distribusi kelas:")
                        st.write(classifier.y_test.value_counts())
    
    # Mode 3: Training Model
    elif app_mode == "Training Model":
        st.header("🎯 Training Model C4.5")
        
        if st.session_state.data is None:
            st.warning("⚠️ Silakan upload dan preprocessing data terlebih dahulu")
            return
        
        if classifier.X_train is None:
            st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu di tab 'Preprocessing'")
            return
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Setting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_depth = st.slider("Max Depth", 1, 20, 5)
        
        with col2:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        
        with col3:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
        
        if st.button("Train Model"):
            with st.spinner("Training model C4.5..."):
                success = classifier.train(max_depth, min_samples_split, min_samples_leaf)
                
                if success:
                    st.session_state.trained = True
                    st.success("✅ Model berhasil ditraining!")
                    
                    # Tampilkan info model
                    st.subheader("Model Information")
                    st.write(f"- Depth: {classifier.model.get_depth()}")
                    st.write(f"- Number of leaves: {classifier.model.get_n_leaves()}")
                    st.write(f"- Number of features: {len(classifier.feature_names)}")
    
    # Mode 4: Evaluasi
    elif app_mode == "Evaluasi":
        st.header("📊 Evaluasi Model")
        
        if not st.session_state.trained:
            st.warning("⚠️ Silakan training model terlebih dahulu di tab 'Training Model'")
            return
        
        # Evaluasi model
        evaluation = classifier.evaluate()
        
        if evaluation is not None:
            # Tampilkan metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Accuracy", f"{evaluation['train_accuracy']:.2%}")
            
            with col2:
                st.metric("Test Accuracy", f"{evaluation['test_accuracy']:.2%}")
            
            # Classification Report
            st.subheader("Classification Report")
            report_df = pd.DataFrame(evaluation['classification_report']).transpose()
            st.dataframe(report_df)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(classifier.y_test, evaluation['y_test_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classifier.class_names,
                       yticklabels=classifier.class_names,
                       ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Visualisasi Decision Tree - BAGIAN YANG DIPERBAIKI
            st.subheader("Visualisasi Decision Tree")
            
            # Slider untuk kedalaman pohon
            tree_depth = st.slider("Kedalaman pohon yang ditampilkan", 2, 10, 5)
            
            try:
                fig, ax = plt.subplots(figsize=(20, 10))
                
                # Konversi feature_names dan class_names ke list - PERBAIKAN UTAMA
                feature_names_list = classifier.feature_names.tolist() 
                class_names_list = classifier.class_names.tolist()
                
                plot_tree(classifier.model,
                         feature_names=feature_names_list,  # SEKARANG SUDAH LIST
                         class_names=class_names_list,      # SEKARANG SUDAH LIST
                         filled=True,
                         rounded=True,
                         max_depth=tree_depth,
                         ax=ax,
                         fontsize=10)
                
                plt.title("Decision Tree - Algoritma C4.5")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error saat menampilkan decision tree: {str(e)}")
                
                # Fallback: tampilkan text representation
                from sklearn.tree import export_text
                try:
                    tree_rules = export_text(classifier.model, 
                                           feature_names=feature_names_list)
                    st.text_area("Decision Tree Rules (Text Version):", tree_rules, height=300)
                except Exception as export_error:
                    st.error(f"Juga gagal menampilkan text version: {str(export_error)}")
            
            # Feature Importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': classifier.feature_names,
                'importance': classifier.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
            ax.set_title('Top 10 Feature Importance')
            st.pyplot(fig)
    
    # Mode 5: Prediksi
    elif app_mode == "Prediksi":
        st.header("🔮 Prediksi Data Baru")
        
        if not st.session_state.trained:
            st.warning("⚠️ Silakan training model terlebih dahulu di tab 'Training Model'")
            return
        
        # Opsi input data baru
        st.subheader("Input Data untuk Prediksi")
        
        # Buat form input berdasarkan feature names
        input_data = {}
        col1, col2 = st.columns(2)
        
        features = classifier.feature_names.tolist()
        half = len(features) // 2
        
        with col1:
            for feature in features[:half]:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        with col2:
            for feature in features[half:]:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("Predict"):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Predict
            predictions = classifier.predict(input_df)
            
            if predictions is not None:
                st.success(f"🎯 Prediksi: **{predictions[0]}**")
                
                # Tampilkan probability jika available
                try:
                    probabilities = classifier.model.predict_proba(input_df)
                    prob_df = pd.DataFrame({
                        'Kelas': classifier.class_names,
                        'Probability': probabilities[0]
                    }).sort_values('Probability', ascending=False)
                    
                    st.subheader("Probabilities")
                    st.dataframe(prob_df)
                    
                    # Visualisasi probabilities
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=prob_df, x='Probability', y='Kelas', ax=ax)
                    ax.set_title('Prediction Probabilities')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.info("Probability tidak tersedia untuk model ini")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Aplikasi Klasifikasi C4.5** • "
        "Dibuat dengan Streamlit dan Scikit-learn"
    )

if __name__ == "__main__":
    main()
