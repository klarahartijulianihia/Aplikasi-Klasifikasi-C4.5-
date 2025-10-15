import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import io
import textwrap

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Status Rumah - C4.5",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .feature-importance {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e6f7e6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2ecc71;
    }
    .conclusion-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
    }
    .recommendation-box {
        background-color: #d1ecf1;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

class C45Classifier:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.class_names = None
        
    def preprocess_data(self, df):
        """Preprocessing data"""
        # Buat copy dataframe
        df_processed = df.copy()
        
        # Preprocessing Gaji
        df_processed['Gaji'] = df_processed['Gaji'].replace('Rp.-', '0')
        df_processed['Gaji'] = df_processed['Gaji'].str.replace('Rp.', '').str.replace('.', '').astype(float)
        
        # Preprocessing Tanggal Lahir dan hitung Usia
        def fix_date(date_str):
            if isinstance(date_str, str):
                if '//' in date_str:
                    parts = date_str.split('//')
                    if len(parts) == 2:
                        day = parts[0]
                        month_year = parts[1].split('/')
                        if len(month_year) == 2:
                            return f"{day}/{month_year[0]}/{month_year[1]}"
                elif '/' in date_str and len(date_str.split('/')) == 3:
                    parts = date_str.split('/')
                    if len(parts[2]) == 4:
                        return date_str
                    else:
                        day = parts[0]
                        month = parts[1][:2]
                        year = parts[1][2:] if len(parts[1]) > 2 else parts[2]
                        return f"{day}/{month}/{year}"
            return date_str
        
        df_processed['Tanggal Lahir'] = df_processed['Tanggal Lahir'].apply(fix_date)
        df_processed['Tanggal Lahir'] = pd.to_datetime(df_processed['Tanggal Lahir'], errors='coerce')
        df_processed['Usia'] = (pd.to_datetime('today') - df_processed['Tanggal Lahir']).dt.days // 365
        
        # Handle missing values
        if df_processed['Usia'].isnull().any():
            df_processed['Usia'] = df_processed['Usia'].fillna(df_processed['Usia'].median())
        
        # Feature Engineering
        # Kategorisasi Gaji
        def categorize_income(income):
            if income == 0:
                return 'Tidak_Berpenghasilan'
            elif income < 3000000:
                return 'Rendah'
            elif income < 5000000:
                return 'Menengah'
            else:
                return 'Tinggi'
        
        df_processed['Kategori_Gaji'] = df_processed['Gaji'].apply(categorize_income)
        
        # Kategorisasi Usia
        def categorize_age(age):
            if age < 30:
                return 'Muda'
            elif age < 50:
                return 'Dewasa'
            else:
                return 'Tua'
        
        df_processed['Kategori_Usia'] = df_processed['Usia'].apply(categorize_age)
        
        # Kategorisasi Jumlah ART
        def categorize_family_size(size):
            if size == 1:
                return 'Tunggal'
            elif size <= 3:
                return 'Kecil'
            elif size <= 5:
                return 'Sedang'
            else:
                return 'Besar'
        
        df_processed['Kategori_ART'] = df_processed['jmlh Art'].apply(categorize_family_size)
        
        return df_processed
    
    def train_model(self, df, target_col, test_size=0.2, max_depth=5):
        """Train C4.5 model"""
        features = ['Kategori_Usia', 'Kategori_ART', 'Kategori_Gaji', 'Pekerjaan']
        
        # Prepare features and target
        X = pd.get_dummies(df[features])
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.feature_names = X.columns
        self.class_names = [str(cls) for cls in np.unique(y)]
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        return X_train, X_test, y_train, y_test, y_pred, X
    
    def evaluate_model(self, y_test, y_pred):
        """Evaluate model performance"""
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        return metrics, cm, report
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def predict_new(self, input_data):
        """Predict new data"""
        if self.model is None:
            return None
        
        return self.model.predict(input_data)

def generate_business_insights(df_processed, metrics, feature_importance, cm, target_col):
    """Generate automated business insights and conclusions"""
    
    insights = []
    conclusions = []
    recommendations = []
    
    # Basic statistics
    total_samples = len(df_processed)
    accuracy = metrics['Accuracy']
    top_features = feature_importance.head(5)['Feature'].tolist()
    
    # 1. PERFORMANCE INSIGHTS
    insights.append("## 📊 **Insight Kinerja Model**")
    
    if accuracy >= 0.8:
        insights.append("✅ **Kinerja Excellent**: Model memiliki akurasi sangat tinggi (>80%) dan dapat diandalkan untuk prediksi")
    elif accuracy >= 0.7:
        insights.append("⚠️ **Kinerja Baik**: Model memiliki akurasi yang memadai untuk kebanyakan kasus bisnis")
    else:
        insights.append("❌ **Kinerja Perlu Perbaikan**: Akurasi model di bawah 70%, pertimbangkan feature engineering atau data tambahan")
    
    # 2. DATA DISTRIBUTION INSIGHTS
    insights.append("\n## 📈 **Insight Distribusi Data**")
    
    # Target distribution
    target_dist = df_processed[target_col].value_counts()
    majority_class = target_dist.idxmax()
    majority_percentage = (target_dist.max() / total_samples) * 100
    
    insights.append(f"🔍 **Distribusi Kelas Target**: Kelas {majority_class} mendominasi ({majority_percentage:.1f}% dari total data)")
    
    if majority_percentage > 70:
        insights.append("⚠️ **Imbalance Warning**: Data tidak seimbang, model mungkin bias terhadap kelas mayoritas")
    
    # Age distribution insights
    age_stats = df_processed['Usia'].describe()
    insights.append(f"👥 **Profil Usia**: Rata-rata usia {age_stats['mean']:.1f} tahun (range: {age_stats['min']:.0f}-{age_stats['max']:.0f} tahun)")
    
    # Income insights
    income_stats = df_processed['Gaji'].describe()
    zero_income_count = (df_processed['Gaji'] == 0).sum()
    insights.append(f"💰 **Profil Pendapatan**: Rata-rata gaji Rp {income_stats['mean']:,.0f} ({zero_income_count} tidak berpenghasilan)")
    
    # 3. FEATURE IMPORTANCE INSIGHTS
    insights.append("\n## 🎯 **Insight Faktor Penentu**")
    
    top_3_features = feature_importance.head(3)
    for idx, row in top_3_features.iterrows():
        feature_name = row['Feature']
        importance = row['Importance']
        insights.append(f"🏆 **Faktor #{idx+1}**: {feature_name} (importance: {importance:.3f})")
    
    # 4. BUSINESS CONCLUSIONS
    conclusions.append("## 🎯 **Kesimpulan Analisis**")
    
    # Model reliability conclusion
    if accuracy >= 0.75:
        conclusions.append(f"✅ **Model dapat diandalkan** dengan akurasi {accuracy:.1%} untuk memprediksi status rumah")
    else:
        conclusions.append(f"⚠️ **Model memerlukan improvement** dengan akurasi saat ini {accuracy:.1%}")
    
    # Key drivers conclusion
    main_driver = top_3_features.iloc[0]['Feature']
    conclusions.append(f"🔑 **Faktor penentu utama**: {main_driver} memiliki pengaruh paling signifikan")
    
    # Data quality conclusion
    conclusions.append(f"📊 **Kualitas data**: {total_samples} sampel memberikan basis data yang {'cukup' if total_samples >= 50 else 'terbatas'} untuk analisis")
    
    # 5. STRATEGIC RECOMMENDATIONS
    recommendations.append("## 💡 **Rekomendasi Bisnis**")
    
    # Based on feature importance
    if any('Gaji' in feature for feature in top_3_features['Feature']):
        recommendations.append("💰 **Fokus ekonomi**: Tingkatkan program bantuan pendapatan karena gaji merupakan faktor penting")
    
    if any('Usia' in feature for feature in top_3_features['Feature']):
        recommendations.append("👵 **Program usia**: Rancang program khusus untuk kelompok usia tertentu berdasarkan analisis")
    
    if any('ART' in feature for feature in top_3_features['Feature']):
        recommendations.append("👨‍👩‍👧‍👦 **Keluarga**: Pertimbangkan ukuran keluarga dalam program bantuan")
    
    # General recommendations
    if accuracy < 0.7:
        recommendations.append("🔄 **Kumpulkan data tambahan** untuk meningkatkan akurasi model")
    
    recommendations.append("📋 **Validasi lapangan**: Lakukan cross-check dengan kondisi aktual di lapangan")
    recommendations.append("🎯 **Segmentasi**: Gunakan model untuk identifikasi kelompok prioritas bantuan")
    
    return insights, conclusions, recommendations

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Klasifikasi Status Rumah - Algoritma C4.5</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Konfigurasi")
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload file Excel", type=['xlsx'])
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Data berhasil dimuat! {df.shape[0]} baris, {df.shape[1]} kolom")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
    else:
        # Use sample data
        st.info("📁 Upload file Excel Anda atau gunakan data sample di bawah")
        # Create sample data display
        sample_data = {
            'No': [1, 2, 3],
            'No_KK': ['1218100025000100', '1218100025000010', '1218100025000250'],
            'Nama_KRT': ['SUHENDRA SIREGAR', 'SUKIMIN', 'SAWIYAH'],
            'Alamat': ['Lingkungan V', 'Lingkungan V', 'Lingkungan V'],
            'Tanggal Lahir': ['1978-03-04', '1987-09-08', '1969-01-16'],
            'jmlh Art': [7, 4, 3],
            'Pekerjaan': ['Wiraswasta', 'Wiraswasta', 'Karyawan Swasta'],
            'Gaji': ['Rp.5.420.000', 'Rp.4.560.000', 'Rp.7.600.000'],
            'status rumah': [1, 1, 1]
        }
        df = pd.DataFrame(sample_data)
        st.warning("Menggunakan data sample. Silakan upload file Excel Anda.")
    
    # Tampilkan data
    st.subheader("📊 Data Preview")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**5 Data Teratas:**")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.write("**Informasi Data:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()
        st.text_area("Data Info", info_text, height=200, disabled=True)
    
    # Preprocessing dan Modeling
    st.subheader("🔧 Preprocessing & Modeling")
    
    # Parameter model
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    with col2:
        max_depth = st.slider("Max Depth", 3, 10, 5)
    with col3:
        target_col = st.selectbox("Target Column", df.columns, index=len(df.columns)-1)
    
    if st.button("🚀 Jalankan Model C4.5", type="primary"):
        with st.spinner("Melakukan preprocessing dan training model..."):
            # Initialize classifier
            classifier = C45Classifier()
            
            # Preprocess data
            df_processed = classifier.preprocess_data(df)
            
            # Train model
            X_train, X_test, y_train, y_test, y_pred, X_full = classifier.train_model(
                df_processed, target_col, test_size, max_depth
            )
            
            # Evaluate model
            metrics, cm, report = classifier.evaluate_model(y_test, y_pred)
            
            # Get feature importance
            feature_importance = classifier.get_feature_importance()
            
            # Generate insights
            insights, conclusions, recommendations = generate_business_insights(
                df_processed, metrics, feature_importance, cm, target_col
            )
            
            # Store in session state
            st.session_state.classifier = classifier
            st.session_state.df_processed = df_processed
            st.session_state.metrics = metrics
            st.session_state.cm = cm
            st.session_state.report = report
            st.session_state.X_full = X_full
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.feature_importance = feature_importance
            st.session_state.insights = insights
            st.session_state.conclusions = conclusions
            st.session_state.recommendations = recommendations
            
            st.success("✅ Model berhasil dilatih dan dievaluasi!")
    
    # Display results if model is trained
    if 'classifier' in st.session_state:
        classifier = st.session_state.classifier
        metrics = st.session_state.metrics
        cm = st.session_state.cm
        report = st.session_state.report
        insights = st.session_state.insights
        conclusions = st.session_state.conclusions
        recommendations = st.session_state.recommendations
        
        # Metrics
        st.subheader("📈 Hasil Evaluasi Model")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=np.unique(st.session_state.y_test),
                   yticklabels=np.unique(st.session_state.y_test))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("📋 Classification Report")
        st.text(report)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        importance_df = classifier.get_feature_importance()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Top 10 Fitur Penting:**")
            st.dataframe(importance_df.head(10), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = importance_df.head(10)
            ax.barh(top_features['Feature'], top_features['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('10 Fitur Terpenting')
            ax.invert_yaxis()
            st.pyplot(fig)
        
        # BUSINESS INSIGHTS SECTION
        st.subheader("💡 Insight Bisnis & Kesimpulan")
        
        # Tabs untuk organisasi yang lebih baik
        tab1, tab2, tab3 = st.tabs(["📊 Insights", "🎯 Kesimpulan", "💡 Rekomendasi"])
        
        with tab1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            for insight in insights:
                lines = textwrap.wrap(insight, width=80)
                for line in lines:
                    st.write(line)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="conclusion-box">', unsafe_allow_html=True)
            for conclusion in conclusions:
                lines = textwrap.wrap(conclusion, width=80)
                for line in lines:
                    st.write(line)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            for recommendation in recommendations:
                lines = textwrap.wrap(recommendation, width=80)
                for line in lines:
                    st.write(line)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Summary Card
        st.subheader("📋 Executive Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Akurasi Model", f"{metrics['Accuracy']:.1%}")
        with col2:
            top_feature = importance_df.iloc[0]['Feature']
            st.metric("🎯 Faktor Utama", top_feature[:20] + "..." if len(top_feature) > 20 else top_feature)
        with col3:
            total_data = len(st.session_state.df_processed)
            st.metric("📊 Total Data", f"{total_data} KK")
        
        # Decision Tree Visualization
        st.subheader("🌳 Visualisasi Pohon Keputusan C4.5")
        
        tree_depth = st.slider("Kedalaman pohon yang ditampilkan", 2, max_depth, min(3, max_depth))
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(classifier.model, 
                 feature_names=classifier.feature_names,
                 class_names=classifier.class_names,
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 max_depth=tree_depth,
                 ax=ax)
        ax.set_title('Pohon Keputusan C4.5', fontsize=16)
        st.pyplot(fig)
        
        # Data Analysis
        st.subheader("📊 Analisis Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribusi target
            fig, ax = plt.subplots(figsize=(8, 6))
            st.session_state.df_processed[target_col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Distribusi Kelas Target')
            ax.set_xlabel('Status Rumah')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)
        
        with col2:
            # Distribusi usia
            fig, ax = plt.subplots(figsize=(8, 6))
            st.session_state.df_processed['Usia'].hist(bins=20, ax=ax)
            ax.set_title('Distribusi Usia')
            ax.set_xlabel('Usia')
            ax.set_ylabel('Frekuensi')
            st.pyplot(fig)
        
        # Prediksi Data Baru
        st.subheader("🔮 Prediksi Data Baru")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                usia = st.selectbox("Kategori Usia", ['Muda', 'Dewasa', 'Tua'])
                pekerjaan = st.selectbox("Pekerjaan", st.session_state.df_processed['Pekerjaan'].unique())
            
            with col2:
                jumlah_art = st.selectbox("Jumlah ART", ['Tunggal', 'Kecil', 'Sedang', 'Besar'])
                kategori_gaji = st.selectbox("Kategori Gaji", ['Tidak_Berpenghasilan', 'Rendah', 'Menengah', 'Tinggi'])
            
            with col3:
                st.write("**Parameter Input:**")
                st.write(f"Usia: {usia}")
                st.write(f"Jumlah ART: {jumlah_art}")
                st.write(f"Gaji: {kategori_gaji}")
                st.write(f"Pekerjaan: {pekerjaan}")
            
            predict_btn = st.form_submit_button("🎯 Prediksi Status Rumah")
            
            if predict_btn:
                # Prepare input data
                input_data = {}
                for feature in st.session_state.X_full.columns:
                    if f"Kategori_Usia_{usia}" in feature:
                        input_data[feature] = 1
                    elif f"Kategori_ART_{jumlah_art}" in feature:
                        input_data[feature] = 1
                    elif f"Kategori_Gaji_{kategori_gaji}" in feature:
                        input_data[feature] = 1
                    elif f"Pekerjaan_{pekerjaan}" in feature:
                        input_data[feature] = 1
                    else:
                        input_data[feature] = 0
                
                input_df = pd.DataFrame([input_data])
                
                # Predict
                prediction = classifier.predict_new(input_df)
                
                st.success(f"**Hasil Prediksi:** Status Rumah = {prediction[0]}")
                
                # Show probabilities
                probabilities = classifier.model.predict_proba(input_df)[0]
                prob_df = pd.DataFrame({
                    'Kelas': classifier.class_names,
                    'Probabilitas': probabilities
                }).sort_values('Probabilitas', ascending=False)
                
                st.write("**Probabilitas per Kelas:**")
                st.dataframe(prob_df, use_container_width=True)
        
        # Download Results
        st.subheader("💾 Download Hasil")
        
        # Create results dataframe
        results_df = st.session_state.df_processed.copy()
        results_df['Prediksi'] = pd.Series(st.session_state.y_pred, index=st.session_state.y_test.index)
        results_df['Aktual'] = st.session_state.y_test
        results_df['Benar'] = results_df['Prediksi'] == results_df['Aktual']
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_status_rumah.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()