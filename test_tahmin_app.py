import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import sys
import time
import json
from datetime import datetime
import io
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

# Başlık ayarla
st.set_page_config(page_title="Test Management AI", layout="wide")

# Yardımcı fonksiyonlar - tüm sayfalar tarafından erişilebilecek şekilde
def generate_new_id(existing_ids=None):
    """Unique bir ID oluşturur, formatı: TC-YYYYMMDD-HHMMSS-NNNN"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_suffix = random.randint(1000, 9999)
    new_id = f"TC-{timestamp}-{random_suffix}"
    
    # Eğer bu ID zaten kullanılmışsa, başka bir rastgele sayı ekle
    if existing_ids and new_id in existing_ids:
        random_suffix = random.randint(1000, 9999)
        new_id = f"TC-{timestamp}-{random_suffix}"
    
    return new_id

# Test case yükleme/kaydetme fonksiyonları
def load_test_cases():
    """Test case'leri JSON dosyasından yükler"""
    if os.path.exists("test_cases.json"):
        with open("test_cases.json", "r") as f:
            return json.load(f)
    return []

def save_test_cases(test_cases):
    """Test case'leri JSON dosyasına kaydeder"""
    with open("test_cases.json", "w") as f:
        json.dump(test_cases, f, indent=4)

# Sidebar oluştur
st.sidebar.title("Test Management AI")
st.sidebar.markdown("---")

# Sayfa seçimi
page = st.sidebar.radio(
    "Select Page:",
    ["Prediction Models", "Test Case Creation"]
)

if page == "Prediction Models":
    # Tüm model seçeneklerini tanımla
    ALL_MODELS = [
        {
            'name': 'Temel Model',
            'file': 'test_tahmin_modeli.pkl',
            'script': ['python', 'basit_model.py'],
            'requirements': ['basit_veri_hazirlama.py'],
            'description': """
            **Temel Model Bilgisi:**
            - Algoritma: RandomForest
            - 100 karar ağacı
            - Temel özellikleri kullanır
            - Eğitim süresi: ~5 saniye
            """
        },
        {
            'name': 'Gelişmiş Model',
            'file': 'gelismis_test_model.pkl',
            'script': ['python', 'model_iyilestirme.py'],
            'requirements': ['test_verileri.csv'],
            'description': """
            **Gelişmiş Model Bilgisi:**
            - GridSearch ile optimize edilmiş
            - Daha yüksek doğruluk
            - Hiper-parametreler ayarlanmış
            - Eğitim süresi: ~30 saniye
            """
        },
        {
            'name': 'Derin Öğrenme Modeli',
            'file': 'derin_test_modeli.h5',
            'script': ['python', 'derin_ogrenme_model.py'],
            'requirements': ['test_verileri.csv', 'tensorflow'],
            'description': """
            **Derin Öğrenme Modeli Bilgisi:**
            - Neural Network kullanır
            - 3 katmanlı mimari
            - Dropout katmanları ile overfitting önleme
            - Eğitim süresi: ~60 saniye
            """
        }
    ]

    # Model durumlarını kontrol et
    for model in ALL_MODELS:
        model['exists'] = os.path.exists(model['file'])

    # Sidebar'da model seçimi
    st.sidebar.markdown("## Model Selection")
    selected_model_name = st.sidebar.radio(
        "Select model to use:",
        options=[model['name'] for model in ALL_MODELS],
        index=0
    )

    # Seçilen model bilgisini bul
    selected_model = next(model for model in ALL_MODELS if model['name'] == selected_model_name)

    # Sidebar'da model bilgisi
    st.sidebar.markdown("---")
    st.sidebar.markdown(selected_model['description'])
    st.sidebar.markdown("---")

    # Model durumu
    if selected_model['exists']:
        st.sidebar.success(f"✅ {selected_model['name']} available")
    else:
        st.sidebar.warning(f"⚠️ {selected_model['name']} not created yet")

    # Modeli yeniden eğit butonu
    if st.sidebar.button("Retrain Model", key="retrain"):
        with st.sidebar:
            with st.spinner(f"Training {selected_model['name']}..."):
                # Veri dosyası yoksa oluştur
                if not os.path.exists('test_verileri.csv'):
                    st.info("Creating data file...")
                    subprocess.run(['python', 'basit_veri_hazirlama.py'], check=True)
                    
                # Gerekli kütüphaneleri kontrol et ve kur
                if 'tensorflow' in selected_model['requirements'] and selected_model['name'] == 'Derin Öğrenme Modeli':
                    try:
                        import tensorflow
                    except ImportError:
                        st.info("Installing TensorFlow, this might take a while...")
                        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow'], check=True)
                
                # Modeli eğit
                st.info(f"Training {selected_model['name']}...")
                result = subprocess.run(selected_model['script'], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success(f"{selected_model['name']} successfully trained!")
                    selected_model['exists'] = True
                    time.sleep(1)  # UI güncelleme için kısa bir gecikme
                    st.experimental_rerun()  # Sayfayı yenile
                else:
                    st.error(f"Model training failed: {result.stderr}")

    # Modeli yükle fonksiyonu
    @st.cache_resource
    def model_yukle(model_info):
        if not model_info['exists']:
            # Modeli otomatik olarak oluştur
            with st.spinner(f"Creating {model_info['name']}..."):
                # Veri dosyası yoksa oluştur
                if not os.path.exists('test_verileri.csv'):
                    subprocess.run(['python', 'basit_veri_hazirlama.py'], check=True)
                
                # Modeli eğit
                subprocess.run(model_info['script'], check=True)
                model_info['exists'] = True
        
        # Modeli yükle
        if model_info['file'].endswith('.pkl'):
            with open(model_info['file'], 'rb') as f:
                return pickle.load(f)
        elif model_info['file'].endswith('.h5'):
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(model_info['file'])
            except ImportError:
                st.error("TensorFlow not installed! Deep Learning Model cannot be used.")
                st.info("To install TensorFlow: pip install tensorflow")
                return None

    # Ana başlık
    st.title('Test Failure Prediction Tool')
    st.markdown(f"**Active Model: {selected_model_name}**")
    st.write('Enter test metrics to predict the probability of test failure.')

    # Modeli yüklemeyi dene
    try:
        model = model_yukle(selected_model)
        if model is None and selected_model['name'] == 'Derin Öğrenme Modeli':
            st.error("Deep Learning Model could not be loaded. Please select another model from the sidebar.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Try retraining the model.")
        st.stop()

    # İki sütunlu düzen
    col1, col2 = st.columns(2)

    # Kullanıcı girişleri - sol sütun
    with col1:
        st.subheader("Test Metrics")
        test_suresi = st.slider('Test Duration (seconds)', 10, 100, 50)
        kod_degisiklik = st.slider('Code Change Percentage', 1, 50, 20)
        test_kapsami = st.slider('Test Coverage (%)', 50, 100, 80)
        onceki_hatalar = st.slider('Previous Errors Count', 0, 20, 5)
        
        # Tahmin butonu
        tahmin_butonu = st.button('Predict', use_container_width=True)

    # Tahmin fonksiyonu (modele göre farklılık gösterebilir)
    def tahmin_yap(model, model_info, veri):
        if model_info['file'].endswith('.pkl'):
            # Scikit-learn modelleri için
            tahmin = model.predict(veri)[0]
            olasilik = model.predict_proba(veri)[0][1]
        elif model_info['file'].endswith('.h5'):
            # TensorFlow modelleri için
            tahmin_array = model.predict(veri.values)
            olasilik = float(tahmin_array[0][0])
            tahmin = 1 if olasilik > 0.5 else 0
        
        return tahmin, olasilik

    # Sonuç gösterimi - sağ sütun
    with col2:
        if tahmin_butonu:
            st.subheader("Prediction Result")
            
            # Yükleniyor göstergesi
            with st.spinner('Making prediction...'):
                veri = pd.DataFrame({
                    'test_suresi': [test_suresi],
                    'kod_degisiklik_yuzde': [kod_degisiklik],
                    'test_kapsamı': [test_kapsami],
                    'onceki_hata_sayisi': [onceki_hatalar]
                })
                
                tahmin, olasilik = tahmin_yap(model, selected_model, veri)
            
            if tahmin == 1:
                st.error(f"❌ Test is likely to fail!")
                st.metric("Failure Probability", f"{olasilik*100:.1f}%")
            else:
                st.success(f"✅ Test is likely to succeed.")
                st.metric("Success Probability", f"{(1-olasilik)*100:.1f}%")
            
            # Gösterge grafiği
            st.subheader("Probability Gauge")
            fig, ax = plt.subplots(figsize=(8, 3))
            
            # Modele göre renk ayarı
            if selected_model['name'] == 'Derin Öğrenme Modeli':
                colors = ["green" if olasilik < 0.4 else "orange" if olasilik < 0.6 else "red"]
            else:
                colors = ["green" if olasilik < 0.3 else "orange" if olasilik < 0.7 else "red"]
                
            gauge = sns.barplot(x=[olasilik], y=['Failure Probability'], ax=ax, palette=colors)
            ax.set_xlim(0, 1)
            for i, bar in enumerate(ax.patches):
                ax.text(bar.get_width()/2, i, f"{olasilik*100:.1f}%", ha='center', va='center', 
                        color='black', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Model karşılaştırma
            if st.checkbox("Compare with other models"):
                st.subheader("Model Comparison")
                
                # Önce diğer modelleri mevcut değilse oluştur
                with st.spinner("Preparing other models..."):
                    for diger_model_info in ALL_MODELS:
                        if diger_model_info['name'] != selected_model['name'] and not diger_model_info['exists']:
                            # Tensorflow modeli yüklenmesi sorun çıkarabilir, atlayalım
                            if diger_model_info['name'] == 'Derin Öğrenme Modeli' and 'tensorflow' in diger_model_info['requirements']:
                                try:
                                    import tensorflow
                                except ImportError:
                                    st.warning(f"TensorFlow required for {diger_model_info['name']}, not installed.")
                                    continue
                            
                            st.info(f"Preparing {diger_model_info['name']}...")
                            try:
                                # Veri dosyası yoksa oluştur
                                if not os.path.exists('test_verileri.csv'):
                                    subprocess.run(['python', 'basit_veri_hazirlama.py'], check=True)
                                
                                # Modeli eğit
                                subprocess.run(diger_model_info['script'], check=True)
                                diger_model_info['exists'] = os.path.exists(diger_model_info['file'])
                            except Exception as e:
                                st.warning(f"Could not create {diger_model_info['name']}: {str(e)}")
                
                # Karşılaştırma sonuçlarını sakla
                karsilastirma_sonuclari = {}
                # Önce aktif modeli ekle
                karsilastirma_sonuclari[selected_model['name']] = olasilik
                
                # Diğer modellerin tahminlerini al
                for diger_model_info in ALL_MODELS:
                    if diger_model_info['exists'] and diger_model_info['name'] != selected_model['name']:
                        with st.spinner(f"Getting predictions from {diger_model_info['name']}..."):
                            try:
                                # Modeli yükle
                                if diger_model_info['file'].endswith('.pkl'):
                                    with open(diger_model_info['file'], 'rb') as f:
                                        diger_model = pickle.load(f)
                                elif diger_model_info['file'].endswith('.h5'):
                                    try:
                                        import tensorflow as tf
                                        diger_model = tf.keras.models.load_model(diger_model_info['file'])
                                    except ImportError:
                                        st.warning(f"TensorFlow required for {diger_model_info['name']}, skipping.")
                                        continue
                                
                                # Tahmin yap
                                diger_tahmin, diger_olasilik = tahmin_yap(diger_model, diger_model_info, veri)
                                karsilastirma_sonuclari[diger_model_info['name']] = diger_olasilik
                                st.success(f"{diger_model_info['name']} prediction: {diger_olasilik*100:.1f}% failure probability")
                            except Exception as e:
                                st.warning(f"Could not get prediction from {diger_model_info['name']}: {str(e)}")
                
                # Karşılaştırma sonuçlarını göster
                if len(karsilastirma_sonuclari) > 1:  # En az 2 model (biri aktif model) olmalı
                    st.info(f"Comparing a total of {len(karsilastirma_sonuclari)} models")
                    
                    # Karşılaştırma grafiği
                    karsilastirma_df = pd.DataFrame({
                        'Model': list(karsilastirma_sonuclari.keys()),
                        'Failure Probability': list(karsilastirma_sonuclari.values())
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = sns.barplot(data=karsilastirma_df, x='Model', y='Failure Probability', ax=ax)
                    
                    # Çubukları renklendir
                    for i, bar in enumerate(ax.patches):
                        val = bar.get_height()
                        if val < 0.3:
                            bar.set_color('green')
                        elif val < 0.7:
                            bar.set_color('orange')
                        else:
                            bar.set_color('red')
                        
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f"{val*100:.1f}%", ha='center', fontweight='bold')
                    
                    plt.ylim(0, 1)
                    plt.title("Model Comparison: Failure Probability")
                    st.pyplot(fig)
                else:
                    st.warning("Not enough models available for comparison. Please train other models as well.")

    # Alt bilgi
    st.markdown("---")
    st.markdown("This application uses machine learning models to predict test failures.")
    st.markdown("⚠️ Note: Deep Learning Model may take longer when trained for the first time.")

elif page == "Test Case Creation":
    st.title("Test Case Creation and Management")
    st.markdown("Create, manage and analyze test cases for your software testing process")
    
    # Test case dosyasını kontrol et ve oluştur
    TEST_CASES_FILE = 'test_cases.json'
    if not os.path.exists(TEST_CASES_FILE):
        with open(TEST_CASES_FILE, 'w') as f:
            json.dump([], f)
    
    # Test case'leri yükle
    def load_test_cases():
        try:
            with open(TEST_CASES_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    
    # Test case'leri kaydet
    def save_test_cases(test_cases):
        with open(TEST_CASES_FILE, 'w') as f:
            json.dump(test_cases, f, indent=4)
    
    # Tabs oluştur
    tab1, tab2, tab3, tab4 = st.tabs(["Create Test Case", "View Test Cases", "Analytics", "Import/Export"])
    
    # Tab 1: Test Case Oluşturma
    with tab1:
        st.header("Create Test Case")
        
        # Test Case'in temel bilgileri
        col_title, col_id = st.columns([3, 1])
        with col_title:
            title = st.text_input("Test Title", placeholder="Enter test case title")
        with col_id:
            id_prefix = st.text_input("ID Prefix", value="TC-", help="Optional prefix for test case ID")
        
        # Organize test case info into multiple columns
        col1, col2 = st.columns(2)
        
        with col1:
            area_path = st.text_input("Area Path", value="TDC-Europe", placeholder="e.g. TDC-Europe")
            state = st.selectbox("State", ["Design", "Ready", "In Progress", "Blocked", "Passed", "Failed", "Closed"])
            
        with col2:
            assigned_to = st.text_input("Assigned To", placeholder="Enter name or email")
            priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"])
        
        # Preconditions
        preconditions = st.text_area("Preconditions", 
                                    placeholder="Enter any prerequisites or setup needed before test execution",
                                    help="Example: Test Server : https://mvwtest1.ltg-emea.com/")
        
        # Test Steps with actions and expected results
        st.subheader("Test Steps")
        st.info("Add steps with actions and expected results")
        
        # Container for test steps
        step_container = st.container()
        
        # Initialize session state for steps if not exists
        if 'test_steps' not in st.session_state:
            st.session_state.test_steps = [{'step_num': 1, 'action': '', 'expected': ''}]
        
        # Function to add a new step
        def add_step():
            new_step_num = len(st.session_state.test_steps) + 1
            st.session_state.test_steps.append({'step_num': new_step_num, 'action': '', 'expected': ''})
        
        # Function to remove a step
        def remove_step(step_index):
            if len(st.session_state.test_steps) > 1:  # Always keep at least one step
                del st.session_state.test_steps[step_index]
                # Renumber remaining steps
                for i, step in enumerate(st.session_state.test_steps):
                    step['step_num'] = i + 1
        
        # Display and edit test steps
        with step_container:
            for i, step in enumerate(st.session_state.test_steps):
                col_num, col_action, col_expected, col_del = st.columns([0.5, 2, 2, 0.3])
                
                with col_num:
                    st.text_input("Step #", value=str(step['step_num']), key=f"step_num_{i}", disabled=True)
                
                with col_action:
                    step['action'] = st.text_area("Step Action", value=step['action'], key=f"action_{i}", 
                                             placeholder="Describe the action to perform",
                                             height=100)
                
                with col_expected:
                    step['expected'] = st.text_area("Expected Result", value=step['expected'], key=f"expected_{i}", 
                                             placeholder="What should happen after the step is performed",
                                             height=100)
                
                with col_del:
                    if st.button("🗑️", key=f"del_{i}"):
                        remove_step(i)
                        st.experimental_rerun()
            
            # Button to add more steps
            if st.button("Add Step"):
                add_step()
                st.experimental_rerun()
        
        # Additional test case details
        col1, col2 = st.columns(2)
        
        with col1:
            automated = st.checkbox("Automated", help="Is this test case automated?")
            test_type = st.selectbox("Test Type", ["Functional", "Performance", "Security", "Usability", "Regression", "Integration"])
        
        with col2:
            module = st.text_input("Module/Component", placeholder="Enter the module or component this test belongs to")
            estimated_duration = st.number_input("Estimated Duration (minutes)", min_value=1, max_value=240, value=15)
        
        # Risk assessment
        st.subheader("Risk Assessment")
        
        col1, col2 = st.columns(2)
        with col1:
            complexity = st.slider("Test Complexity", 1, 10, 5, help="1 = Very Simple, 10 = Extremely Complex")
            code_coverage = st.slider("Code Coverage Impact", 1, 10, 5, help="1 = Minimal Coverage, 10 = Critical Coverage")
        
        with col2:
            business_impact = st.slider("Business Impact", 1, 10, 5, help="1 = Low Impact, 10 = High Business Impact")
            failure_history = st.slider("Failure History", 1, 10, 5, help="1 = Never Failed, 10 = Frequently Fails")
        
        # Create test case button
        if st.button("Create Test Case", use_container_width=True):
            if not title:
                st.error("Test Title is required!")
            elif not st.session_state.test_steps[0]['action']:
                st.error("At least one test step action is required!")
            else:
                # Format steps for storage
                formatted_steps = ""
                formatted_expected = ""
                
                for step in st.session_state.test_steps:
                    if step['action']:
                        formatted_steps += f"{step['step_num']}. {step['action']}\n"
                    if step['expected']:
                        formatted_expected += f"{step['step_num']}. {step['expected']}\n"
                
                # Calculate risk score based on risk factors
                risk_score = (
                    complexity * 0.3 + 
                    code_coverage * 0.2 + 
                    business_impact * 0.3 + 
                    failure_history * 0.2
                )
                
                # Create new test case
                new_test_case = {
                    'id': generate_new_id(),
                    'title': title,
                    'area_path': area_path,
                    'assigned_to': assigned_to,
                    'state': state,
                    'priority': priority,
                    'preconditions': preconditions,
                    'steps': formatted_steps,
                    'expected_result': formatted_expected,
                    'type': test_type,
                    'module': module,
                    'complexity': complexity,
                    'code_coverage': code_coverage,
                    'business_impact': business_impact,
                    'failure_history': failure_history,
                    'risk_score': risk_score,
                    'automated': automated,
                    'estimated_duration': estimated_duration,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Load existing test cases
                test_cases = load_test_cases()
                
                # Add new test case
                test_cases.append(new_test_case)
                
                # Save updated test cases
                save_test_cases(test_cases)
                
                # Success message
                st.success("Test Case created successfully!")
                
                # Clear form
                st.session_state.test_steps = [{'step_num': 1, 'action': '', 'expected': ''}]
                
                # Redirect to view tab
                st.info("Go to 'View Test Cases' tab to see your new test case.")
    
    # Tab 2: Test case listesi
    with tab2:
        st.header("View Test Cases")
        
        # Test case'leri yükle
        test_cases = load_test_cases()
        
        # Yinelenen ID'leri düzelt
        if test_cases:
            seen_ids = set()
            ids_to_update = {}
            
            # Yinelenen ID'leri tespit et
            for i, tc in enumerate(test_cases):
                tc_id = tc.get('id', '')
                if tc_id in seen_ids:
                    # Yeni bir ID oluştur
                    new_id = generate_new_id(seen_ids)
                    ids_to_update[i] = new_id
                else:
                    seen_ids.add(tc_id)
            
            # Yinelenen ID'leri güncelle
            if ids_to_update:
                for idx, new_id in ids_to_update.items():
                    st.warning(f"Fixed duplicate ID: {test_cases[idx]['id']} → {new_id}")
                    test_cases[idx]['id'] = new_id
                # Güncellenmiş test senaryolarını kaydet
                save_test_cases(test_cases)
                st.success("Fixed duplicate IDs and saved changes!")
        
        if not test_cases:
            st.info("No test cases found. Create some test cases in the 'Create Test Case' tab!")
        else:
            # Filter and search
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_term = st.text_input("Search", placeholder="Search in titles and descriptions...")
            
            with col2:
                filter_by = st.selectbox("Filter by", ["All", "Priority", "Type", "Module", "Risk Level"])
            
            if filter_by != "All":
                if filter_by == "Priority":
                    filter_value = st.selectbox("Select Priority", ["All"] + sorted(list(set([tc["priority"] for tc in test_cases]))))
                elif filter_by == "Type":
                    filter_value = st.selectbox("Select Type", ["All"] + sorted(list(set([tc["type"] for tc in test_cases]))))
                elif filter_by == "Module":
                    filter_value = st.selectbox("Select Module", ["All"] + sorted(list(set([tc["module"] for tc in test_cases if tc["module"]]))))
                elif filter_by == "Risk Level":
                    filter_value = st.selectbox("Select Risk Level", ["All", "Low", "Medium", "High"])
            
            # Apply filters
            filtered_cases = test_cases
            
            if search_term:
                filtered_cases = [tc for tc in filtered_cases if search_term.lower() in tc["title"].lower() or 
                                 (tc["description"] and search_term.lower() in tc["description"].lower())]
            
            if filter_by != "All" and filter_value != "All":
                if filter_by == "Risk Level":
                    if filter_value == "Low":
                        filtered_cases = [tc for tc in filtered_cases if tc["risk_score"] < 4]
                    elif filter_value == "Medium":
                        filtered_cases = [tc for tc in filtered_cases if 4 <= tc["risk_score"] < 7]
                    else:  # High
                        filtered_cases = [tc for tc in filtered_cases if tc["risk_score"] >= 7]
                else:
                    field = filter_by.lower()
                    filtered_cases = [tc for tc in filtered_cases if tc.get(field) == filter_value]
            
            st.write(f"Showing {len(filtered_cases)} of {len(test_cases)} test cases")
            
            # Toplu silme işlemleri
            col_bulk1, col_bulk2 = st.columns([3, 1])
            with col_bulk1:
                bulk_action = st.selectbox(
                    "Bulk Actions:",
                    ["Select Action", "Delete Selected", "Mark as Passed", "Mark as Failed", "Mark as Blocked"]
                )
            with col_bulk2:
                if bulk_action != "Select Action":
                    if st.button("Apply", use_container_width=True):
                        if bulk_action == "Delete Selected" and len(test_case_selections) > 0:
                            # Silme onayı
                            if st.checkbox(f"Confirm deletion of {len(test_case_selections)} test cases?"):
                                # Seçili test case'leri sil
                                original_count = len(test_cases)
                                test_cases = [tc for tc in test_cases if tc['id'] not in test_case_selections]
                                save_test_cases(test_cases)
                                st.success(f"{original_count - len(test_cases)} test cases deleted successfully!")
                                st.experimental_rerun()
                        elif bulk_action in ["Mark as Passed", "Mark as Failed", "Mark as Blocked"] and len(test_case_selections) > 0:
                            # Test durumlarını güncelle
                            new_state = bulk_action.replace("Mark as ", "")
                            for tc in test_cases:
                                if tc['id'] in test_case_selections:
                                    tc['state'] = new_state
                            save_test_cases(test_cases)
                            st.success(f"{len(test_case_selections)} test cases marked as {new_state}!")
                            st.experimental_rerun()

            # Test case seçim işlemi için session state oluştur
            if 'test_case_selections' not in st.session_state:
                st.session_state.test_case_selections = set()
            
            test_case_selections = st.session_state.test_case_selections

            # Tüm test senaryolarını seçme/seçimi kaldırma
            col_select_all, col_clear = st.columns([3, 1])
            with col_select_all:
                select_all = st.checkbox("Select All Test Cases", key="select_all_checkbox")
                if select_all:
                    test_case_selections = {tc['id'] for tc in filtered_cases}
                else:
                    # Yalnızca "Select All" değiştiğinde seçimi temizle
                    if st.session_state.get('select_all_checkbox') != select_all:
                        test_case_selections = set()

            with col_clear:
                if st.button("Clear Selection", use_container_width=True):
                    test_case_selections = set()

            # Seçili test durumlarını göster
            if test_case_selections:
                st.info(f"{len(test_case_selections)} test case(s) selected")

            # Session state'e kaydet
            st.session_state.test_case_selections = test_case_selections

            # Test case liste görünümünü değiştir (grid/list görünümü ekleyin)
            view_mode = st.radio("View Mode:", ["Detailed", "Grid"], horizontal=True)
            
            if view_mode == "Detailed":
                # Detaylı görünüm
                for tc in filtered_cases:
                    col_select, col_info = st.columns([0.1, 3.9])
                    
                    # Seçim kutusu ekle
                    with col_select:
                        is_selected = st.checkbox("", value=tc['id'] in test_case_selections, key=f"select_{tc['id']}")
                        if is_selected:
                            test_case_selections.add(tc['id'])
                        elif tc['id'] in test_case_selections:
                            test_case_selections.remove(tc['id'])
                    
                    # Test case bilgilerini göster
                    with col_info:
                        with st.expander(f"{tc['id']} - {tc['title']} ({tc.get('priority', 'Medium')})"):
                            # Test case details in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Area Path:** {tc.get('area_path', 'N/A')}")
                                st.markdown(f"**Assigned To:** {tc.get('assigned_to', 'Unassigned')}")
                            with col2:
                                st.markdown(f"**State:** {tc.get('state', 'Design')}")
                                st.markdown(f"**Type:** {tc.get('type', 'N/A')}")
                            with col3:
                                st.markdown(f"**Created:** {tc.get('created_at', 'N/A')}")
                                st.markdown(f"**Est. Duration:** {tc.get('estimated_duration', 'N/A')} min")
                            
                            # Preconditions
                            if tc.get('preconditions'):
                                st.markdown("**Preconditions:**")
                                st.info(tc['preconditions'])
                            
                            # Test steps in a table
                            if tc.get('steps'):
                                # Parse steps and expected results
                                steps_lines = tc['steps'].strip().split('\n')
                                expected_lines = tc.get('expected_result', '').strip().split('\n')
                                
                                # Create a table with step number, action and expected result
                                steps_data = []
                                for i, step in enumerate(steps_lines):
                                    if step.strip():
                                        step_parts = step.split('. ', 1)
                                        step_num = step_parts[0] if len(step_parts) > 1 else str(i+1)
                                        step_action = step_parts[1] if len(step_parts) > 1 else step
                                        
                                        # Find matching expected result
                                        expected = ""
                                        for exp in expected_lines:
                                            if exp.startswith(f"{step_num}."):
                                                expected = exp.split('. ', 1)[1] if len(exp.split('. ', 1)) > 1 else exp
                                                break
                                        
                                        steps_data.append([step_num, step_action, expected])
                                
                                # Display steps table
                                if steps_data:
                                    df = pd.DataFrame(steps_data, columns=["Step #", "Action", "Expected Result"])
                                    st.table(df)
                            
                            # Action buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Edit", key=f"edit_{tc['id']}"):
                                    # TODO: Add edit functionality
                                    st.info("Edit functionality will be implemented in a future update")
                            with col2:
                                # Mevcut silme düğmesini değiştirin
                                if st.button("Delete", key=f"delete_{tc['id']}"):
                                    if st.checkbox(f"Confirm deletion of test case {tc['id']}", key=f"confirm_{tc['id']}"):
                                        test_cases.remove(tc)
                                        save_test_cases(test_cases)
                                        st.success("Test case deleted successfully!")
                                        st.experimental_rerun()
                            with col3:
                                # Durumu değiştirmek için düğme
                                new_state = st.selectbox("Change State", 
                                                        ["Design", "Ready", "In Progress", "Blocked", "Passed", "Failed", "Closed"],
                                                        index=["Design", "Ready", "In Progress", "Blocked", "Passed", "Failed", "Closed"].index(tc.get('state', 'Design')),
                                                        key=f"state_{tc['id']}")
                                if new_state != tc.get('state', 'Design'):
                                    if st.button("Update", key=f"update_{tc['id']}"):
                                        for test_case in test_cases:
                                            if test_case['id'] == tc['id']:
                                                test_case['state'] = new_state
                                                break
                                        save_test_cases(test_cases)
                                        st.success(f"Test case state updated to {new_state}")
                                        st.experimental_rerun()
            else:
                # Grid görünümü - kompakt liste
                # Grid için sütun tanımlamaları
                grid_cols = st.columns([0.1, 0.2, 2, 0.8, 0.8, 0.8])
                grid_cols[0].write("Select")
                grid_cols[1].write("ID")
                grid_cols[2].write("Title")
                grid_cols[3].write("Priority")
                grid_cols[4].write("State")
                grid_cols[5].write("Type")
                
                for tc in filtered_cases:
                    cols = st.columns([0.1, 0.2, 2, 0.8, 0.8, 0.8])
                    
                    # Seçim kutusu
                    is_selected = cols[0].checkbox("", value=tc['id'] in test_case_selections, key=f"grid_{tc['id']}")
                    if is_selected:
                        test_case_selections.add(tc['id'])
                    elif tc['id'] in test_case_selections:
                        test_case_selections.remove(tc['id'])
                        
                    # Temel bilgiler
                    cols[1].write(tc['id'].split('-')[-1] if '-' in tc['id'] else tc['id'])  # ID'nin son kısmını göster
                    cols[2].write(tc['title'])
                    cols[3].write(tc.get('priority', 'Medium'))
                    cols[4].write(tc.get('state', 'Design'))
                    cols[5].write(tc.get('type', 'Functional'))
    
    # Tab 3: Analytics
    with tab3:
        st.header("Test Case Analytics")
        
        # Load test cases
        test_cases = load_test_cases()
        
        if not test_cases:
            st.info("No test cases found. Create some test cases in the 'Create Test Case' tab!")
        else:
            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Test Cases", len(test_cases))
            
            with col2:
                automated_count = sum(1 for tc in test_cases if tc.get('automated', False))
                st.metric("Automated Tests", f"{automated_count} ({automated_count/len(test_cases)*100:.1f}%)")
            
            with col3:
                high_risk = sum(1 for tc in test_cases if tc['risk_score'] >= 7)
                st.metric("High Risk Tests", f"{high_risk} ({high_risk/len(test_cases)*100:.1f}%)")
            
            with col4:
                avg_duration = sum(tc.get('estimated_duration', 0) for tc in test_cases) / len(test_cases)
                st.metric("Avg. Duration", f"{avg_duration:.1f} min")
            
            # Charts
            st.subheader("Test Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Priority distribution
                priority_counts = {priority: 0 for priority in ["Low", "Medium", "High", "Critical"]}
                for tc in test_cases:
                    priority_counts[tc['priority']] += 1
                
                priority_df = pd.DataFrame({
                    'Priority': list(priority_counts.keys()),
                    'Count': list(priority_counts.values())
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(data=priority_df, x='Priority', y='Count', ax=ax, palette=['green', 'blue', 'orange', 'red'])
                
                for i, bar in enumerate(ax.patches):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            f"{bar.get_height():.0f}", ha='center', fontweight='bold')
                
                plt.title("Test Cases by Priority")
                st.pyplot(fig)
            
            with col2:
                # Type distribution
                type_counts = {}
                for tc in test_cases:
                    tc_type = tc['type']
                    if tc_type in type_counts:
                        type_counts[tc_type] += 1
                    else:
                        type_counts[tc_type] = 1
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title("Test Cases by Type")
                st.pyplot(fig)
            
            # Risk distribution
            st.subheader("Risk Distribution")
            
            risk_scores = [tc['risk_score'] for tc in test_cases]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(risk_scores, bins=10, kde=True, ax=ax)
            plt.axvline(x=4, color='green', linestyle='--')
            plt.axvline(x=7, color='red', linestyle='--')
            plt.text(2, plt.ylim()[1]*0.9, "Low Risk", color='green', fontweight='bold')
            plt.text(5.5, plt.ylim()[1]*0.9, "Medium Risk", color='orange', fontweight='bold')
            plt.text(8, plt.ylim()[1]*0.9, "High Risk", color='red', fontweight='bold')
            plt.title("Risk Score Distribution")
            plt.xlabel("Risk Score")
            plt.ylabel("Number of Test Cases")
            st.pyplot(fig)
            
            # Risk factors correlation
            st.subheader("Risk Factors Correlation")
            
            risk_df = pd.DataFrame([{
                'Complexity': tc['complexity'],
                'Code Coverage': tc['code_coverage'],
                'Business Impact': tc['business_impact'],
                'Failure History': tc['failure_history'],
                'Risk Score': tc['risk_score']
            } for tc in test_cases])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(risk_df.corr(), annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
            plt.title("Correlation between Risk Factors")
            st.pyplot(fig)
            
            # Export options
            st.subheader("Export Test Cases")
            
            export_format = st.radio("Export Format", ["JSON", "CSV", "Excel"])
            
            if st.button("Export Test Cases"):
                if export_format == "JSON":
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(test_cases, indent=4),
                        file_name="test_cases_export.json",
                        mime="application/json"
                    )
                elif export_format == "CSV":
                    # Flatten test cases for CSV
                    flat_cases = []
                    for tc in test_cases:
                        flat_tc = tc.copy()
                        if 'steps' in flat_tc:
                            flat_tc['steps'] = flat_tc['steps'].replace('\n', ' ')
                        if 'expected_result' in flat_tc:
                            flat_tc['expected_result'] = flat_tc['expected_result'].replace('\n', ' ')
                        if 'preconditions' in flat_tc:
                            flat_tc['preconditions'] = flat_tc['preconditions'].replace('\n', ' ')
                        flat_cases.append(flat_tc)
                    
                    csv_df = pd.DataFrame(flat_cases)
                    csv = csv_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="test_cases_export.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    st.warning("Excel export requires additional libraries. Please install openpyxl with: pip install openpyxl")
                    try:
                        # Flatten test cases for Excel
                        flat_cases = []
                        for tc in test_cases:
                            flat_tc = tc.copy()
                            if 'steps' in flat_tc:
                                flat_tc['steps'] = flat_tc['steps'].replace('\n', ' ')
                            if 'expected_result' in flat_tc:
                                flat_tc['expected_result'] = flat_tc['expected_result'].replace('\n', ' ')
                            if 'preconditions' in flat_tc:
                                flat_tc['preconditions'] = flat_tc['preconditions'].replace('\n', ' ')
                            flat_cases.append(flat_tc)
                        
                        excel_df = pd.DataFrame(flat_cases)
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer) as writer:
                            excel_df.to_excel(writer, index=False)
                        
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name="test_cases_export.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Excel export failed: {e}")
    
    # Tab 4: Import/Export Test Cases
    with tab4:
        st.header("Import/Export Test Cases")
        
        delete_tab = st.radio("Select Operation", ["Import from CSV", "Excel to CSV Converter", "Export to File", "Delete All Test Cases"], horizontal=True)
        
        if delete_tab == "Delete All Test Cases":
            st.subheader("Delete All Test Cases")
            st.warning("⚠️ This will permanently delete ALL test cases. This action cannot be undone!")
            
            if st.button("Delete All Test Cases", key="delete_all_confirm"):
                # Boş bir liste oluştur ve kaydet
                save_test_cases([])
                # test_cases.json dosyası boş bir dizi içerecek
                st.success("All test cases have been deleted!")
                st.info("The test case database has been reset.")
        
        elif delete_tab == "Import from CSV":
            st.subheader("Import Test Cases from CSV")
            
            st.info("""
            Upload a CSV file with your test cases. The file should contain columns for test case details.
            You can map the columns after uploading the file.
            """)
            
            # CSV dosya yükleyici
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                # CSV'yi yükle
                try:
                    csv_data = pd.read_csv(uploaded_file)
                    st.success(f"CSV file loaded successfully with {len(csv_data)} rows and {len(csv_data.columns)} columns")
                    
                    # CSV önizleme
                    st.subheader("CSV Preview")
                    st.dataframe(csv_data.head(5))
                    
                    # Alan eşleme
                    st.subheader("Map CSV Columns to Test Case Fields")
                    
                    # Mevcut test case alanları
                    required_fields = ['title', 'description', 'priority', 'type', 'steps', 'expected_result']
                    optional_fields = ['module', 'preconditions', 'automated', 'estimated_duration', 'author']
                    risk_fields = ['complexity', 'code_coverage', 'business_impact', 'failure_history']
                    
                    # CSV sütunları
                    csv_columns = list(csv_data.columns)
                    none_option = "-- None --"
                    
                    # Eşleme seçenekleri
                    st.write("**Required Fields**")
                    field_mapping = {}
                    
                    for field in required_fields:
                        # Her bir gerekli alanı için, CSV sütunlarından bir tanesini seçmesini istiyoruz
                        # Eğer alan adı tam olarak eşleşen bir sütun varsa, onu varsayılan olarak seçelim
                        default_index = csv_columns.index(field) if field in csv_columns else 0
                        field_mapping[field] = st.selectbox(
                            f"Map '{field}' to:", 
                            options=[none_option] + csv_columns,
                            index=default_index + 1 if field in csv_columns else 0
                        )
                    
                    st.write("**Optional Fields**")
                    for field in optional_fields:
                        default_index = csv_columns.index(field) if field in csv_columns else 0
                        field_mapping[field] = st.selectbox(
                            f"Map '{field}' to:", 
                            options=[none_option] + csv_columns,
                            index=default_index + 1 if field in csv_columns else 0
                        )
                    
                    st.write("**Risk Assessment Fields**")
                    for field in risk_fields:
                        default_index = csv_columns.index(field) if field in csv_columns else 0
                        field_mapping[field] = st.selectbox(
                            f"Map '{field}' to:", 
                            options=[none_option] + csv_columns,
                            index=default_index + 1 if field in csv_columns else 0
                        )
                    
                    # İçe aktarma butonu
                    if st.button("Import Test Cases"):
                        # Mevcut test case'leri yükle
                        existing_test_cases = load_test_cases()
                        
                        # Yeni ID formatını oluştur
                        def generate_new_id(existing_ids=None):
                            """Unique bir ID oluşturur, formatı: TC-YYYYMMDD-HHMMSS-NNNN"""
                            import random
                            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                            random_suffix = random.randint(1000, 9999)
                            new_id = f"TC-{timestamp}-{random_suffix}"
                            
                            # Eğer bu ID zaten kullanılmışsa, başka bir rastgele sayı ekle
                            if existing_ids and new_id in existing_ids:
                                random_suffix = random.randint(1000, 9999)
                                new_id = f"TC-{timestamp}-{random_suffix}"
                            
                            return new_id

                        # Mevcut ID'leri topla
                        existing_ids = set()
                        if existing_test_cases:
                            existing_ids = {tc.get('id', '') for tc in existing_test_cases}
                        
                        # Her CSV satırını bir test case'e dönüştür
                        imported_cases = []
                        for _, row in csv_data.iterrows():
                            test_case = {}
                            
                            # Required fields
                            missing_required = False
                            for field in required_fields:
                                if field_mapping[field] != none_option:
                                    test_case[field] = str(row[field_mapping[field]])
                                else:
                                    missing_required = True
                                    st.warning(f"Missing required field: {field}")
                                    break
                            
                            if missing_required:
                                continue
                            
                            # Optional fields
                            for field in optional_fields:
                                if field_mapping[field] != none_option:
                                    test_case[field] = str(row[field_mapping[field]])
                            
                            # Risk fields
                            for field in risk_fields:
                                if field_mapping[field] != none_option:
                                    try:
                                        test_case[field] = float(row[field_mapping[field]])
                                    except:
                                        test_case[field] = 5  # Default value if conversion fails
                            
                            # Yeni benzersiz ID oluştur
                            test_case['id'] = generate_new_id(existing_ids)
                            existing_ids.add(test_case['id'])  # Yeni ID'yi listeye ekle
                            test_case['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            
                            # Eğer tüm risk faktörleri varsa risk puanını hesapla, yoksa ortalama değer kullan
                            if all(field in test_case for field in risk_fields):
                                test_case['risk_score'] = (
                                    test_case['complexity'] * 0.3 + 
                                    test_case['code_coverage'] * 0.2 + 
                                    test_case['business_impact'] * 0.3 + 
                                    test_case['failure_history'] * 0.2
                                )
                            else:
                                # Eksik risk faktörleri için varsayılan değerler ata
                                for field in risk_fields:
                                    if field not in test_case:
                                        test_case[field] = 5  # Orta seviye risk
                                test_case['risk_score'] = 5  # Orta seviye risk
                            
                            imported_cases.append(test_case)
                        
                        # Test case'leri mevcut olanlarla birleştir
                        all_test_cases = existing_test_cases + imported_cases
                        
                        # Kaydet
                        save_test_cases(all_test_cases)
                        
                        st.success(f"Successfully imported {len(imported_cases)} test cases!")
                        st.info("Go to 'View Test Cases' tab to see the imported test cases.")
                        
                except Exception as e:
                    st.error(f"Error importing CSV: {str(e)}")
                    st.info("Please make sure your CSV file is correctly formatted.")
        
        elif delete_tab == "Excel to CSV Converter":
            st.subheader("Excel to Test Case CSV Converter")
            
            st.info("""
            Upload an Excel file with your test cases. This tool will:
            1. Automatically map columns to test case fields
            2. Use AI to determine missing fields (priority, type, etc.)
            3. Format the output as a compatible CSV file
            """)
            
            # Excel dosyasını yükle
            uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
            
            if uploaded_file is not None:
                try:
                    # Excel dosyasını oku
                    df = pd.read_excel(uploaded_file)
                    
                    # Önizleme göster
                    st.subheader("Excel Preview")
                    st.dataframe(df.head(5))
                    
                    # İşlenecek sütunları seç
                    st.subheader("Column Selection")
                    
                    all_columns = df.columns.tolist()
                    selected_columns = st.multiselect(
                        "Select columns to include (deselect any columns like 'Work Item' you want to ignore)",
                        all_columns,
                        default=all_columns
                    )
                    
                    # Sütun eşleştirme
                    st.subheader("Column Mapping")
                    
                    # Hedef sütunlar
                    target_columns = [
                        'title', 'description', 'preconditions', 'steps', 
                        'expected_result', 'priority', 'type', 'module'
                    ]
                    
                    # Otomatik sütun eşleştirme
                    column_mapping = {}
                    
                    # Sütun adları benzerliğine göre otomatik eşleştirme
                    column_similarity = {
                        'title': ['title', 'name', 'test case', 'test name', 'summary'],
                        'description': ['description', 'desc', 'summary', 'overview', 'details'],
                        'preconditions': ['precondition', 'pre-condition', 'prerequisites', 'pre condition', 'setup'],
                        'steps': ['steps', 'test steps', 'step', 'procedure', 'test procedure', 'action'],
                        'expected_result': ['expected result', 'expected', 'result', 'outcome', 'expected outcome', 'verification'],
                        'priority': ['priority', 'prio', 'importance', 'severity'],
                        'type': ['type', 'test type', 'category', 'test category'],
                        'module': ['module', 'component', 'feature', 'area', 'function']
                    }
                    
                    # Her bir hedef sütun için en iyi eşleşmeyi bul
                    for target in target_columns:
                        best_match = None
                        for col in selected_columns:
                            col_lower = col.lower()
                            # Tam eşleşme
                            if col_lower == target:
                                best_match = col
                                break
                            # Yaklaşık eşleşme
                            for keyword in column_similarity[target]:
                                if keyword in col_lower:
                                    best_match = col
                                    break
                            if best_match:
                                break
                        
                        # Kullanıcı onayı için varsayılan değeri ayarla
                        column_mapping[target] = st.selectbox(
                            f"Map '{target}' to:",
                            options=["-- Generate with AI --"] + selected_columns,
                            index=0 if not best_match else selected_columns.index(best_match) + 1
                        )
                    
                    # AI ile içerik analizi ve eksik sütunları doldurma
                    st.subheader("AI Content Analysis")
                    
                    ai_settings = st.expander("AI Generation Settings", expanded=False)
                    
                    with ai_settings:
                        priority_keywords = st.text_area(
                            "Priority Classification Keywords",
                            "Critical: critical, blocker, showstopper, urgent, highest\n"
                            "High: high, important, major, significant\n"
                            "Medium: medium, moderate, normal, average\n"
                            "Low: low, minor, trivial, cosmetic",
                            height=100
                        )
                        
                        type_keywords = st.text_area(
                            "Test Type Classification Keywords",
                            "Functional: functionality, feature, function, operation, behavior\n"
                            "Performance: performance, speed, load, stress, response time, efficiency\n"
                            "Security: security, vulnerability, authentication, authorization, encryption\n"
                            "Usability: usability, user experience, ui, ux, interface, ease of use\n"
                            "Regression: regression, existing, previously working\n"
                            "Integration: integration, component, modules, interface, connection",
                            height=150
                        )
                    
                    # İşleme butonu
                    if st.button("Process Excel and Convert to CSV"):
                        with st.spinner("Processing Excel file with AI..."):
                            # Alanları yeni DataFrame'e kopyala
                            result_df = pd.DataFrame()
                            
                            # Seçilen sütunları eşleştir
                            for target in target_columns:
                                if column_mapping[target] != "-- Generate with AI --":
                                    result_df[target] = df[column_mapping[target]]
                                else:
                                    # AI ile eksik sütunu doldur
                                    result_df[target] = None
                            
                            # Priority sütunu yok veya AI ile oluşturulacaksa
                            if column_mapping['priority'] == "-- Generate with AI --":
                                st.info("Generating priority values with AI...")
                                
                                # Öncelik sınıflandırıcısı
                                priority_levels = ["Critical", "High", "Medium", "Low"]
                                
                                # Anahtar kelimeleri ayır
                                priority_dict = {}
                                for line in priority_keywords.strip().split('\n'):
                                    level, keywords = line.split(':', 1)
                                    priority_dict[level.strip()] = [k.strip() for k in keywords.split(',')]
                                
                                # Her test için öncelik belirle
                                priorities = []
                                for idx, row in df.iterrows():
                                    # Analiz edilecek içeriği birleştir
                                    content = ""
                                    for col in selected_columns:
                                        if pd.notna(row[col]):
                                            content += " " + str(row[col])
                                    
                                    content = content.lower()
                                    # Anahtar kelime sayısına göre öncelik belirle
                                    scores = {level: 0 for level in priority_levels}
                                    
                                    for level, keywords in priority_dict.items():
                                        for keyword in keywords:
                                            if keyword.lower() in content:
                                                scores[level] = scores.get(level, 0) + 1
                                    
                                    # En çok puan alan seviyeyi belirle
                                    max_score = 0
                                    priority = "Medium"  # Varsayılan
                                    
                                    for level, score in scores.items():
                                        if score > max_score:
                                            max_score = score
                                            priority = level
                                    
                                    priorities.append(priority)
                                
                                result_df['priority'] = priorities
                            
                            # Type sütunu yok veya AI ile oluşturulacaksa
                            if column_mapping['type'] == "-- Generate with AI --":
                                st.info("Generating test type values with AI...")
                                
                                # Test türü sınıflandırıcısı
                                test_types = ["Functional", "Performance", "Security", "Usability", "Regression", "Integration"]
                                
                                # Anahtar kelimeleri ayır
                                type_dict = {}
                                for line in type_keywords.strip().split('\n'):
                                    type_name, keywords = line.split(':', 1)
                                    type_dict[type_name.strip()] = [k.strip() for k in keywords.split(',')]
                                
                                # Her test için tür belirle
                                types = []
                                for idx, row in df.iterrows():
                                    # Analiz edilecek içeriği birleştir
                                    content = ""
                                    for col in selected_columns:
                                        if pd.notna(row[col]):
                                            content += " " + str(row[col])
                                    
                                    content = content.lower()
                                    # Anahtar kelime sayısına göre tür belirle
                                    scores = {test_type: 0 for test_type in test_types}
                                    
                                    for test_type, keywords in type_dict.items():
                                        for keyword in keywords:
                                            if keyword.lower() in content:
                                                scores[test_type] = scores.get(test_type, 0) + 1
                                    
                                    # En çok puan alan türü belirle
                                    max_score = 0
                                    type_value = "Functional"  # Varsayılan
                                    
                                    for test_type, score in scores.items():
                                        if score > max_score:
                                            max_score = score
                                            type_value = test_type
                                    
                                    types.append(type_value)
                                
                                result_df['type'] = types
                            
                            # Eksik diğer sütunlar için varsayılan değerler
                            for col in target_columns:
                                if col not in result_df.columns or result_df[col].isna().all():
                                    if col == 'module':
                                        result_df[col] = "General"
                                    elif col == 'description' and 'title' in result_df.columns:
                                        result_df[col] = "Test case for " + result_df['title']
                                    elif col not in ['priority', 'type']:  # Bu sütunlar zaten işlendi
                                        result_df[col] = ""
                            
                            # AI işleme sonuçlarını göster
                            st.subheader("Processed Result")
                            st.dataframe(result_df)
                            
                            # CSV'ye dönüştür
                            csv = result_df.to_csv(index=False)
                            
                            # İndirme butonu
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="test_cases_processed.csv",
                                mime="text/csv"
                            )
                            
                            # Sonuçları açıkla
                            st.success("Excel file processed successfully!")
                            st.info("""
                            **AI Processing Results:**
                            
                            - Priority values were determined based on keywords and content analysis
                            - Test types were classified based on test description and steps
                            - All required fields have been filled or generated
                            
                            You can now import this CSV file using the 'Import from CSV' option.
                            """)
                            
                except Exception as e:
                    st.error(f"Error processing Excel file: {str(e)}")
                    st.info("Please make sure your Excel file is properly formatted.")
        
        else:  # Export to File
            # Mevcut dışa aktarma kodunu buraya taşıyın
            st.subheader("Export Test Cases")
            
            # Load test cases
            test_cases = load_test_cases()
            
            if not test_cases:
                st.info("No test cases to export. Create some test cases first!")
            else:
                export_format = st.radio("Export Format", ["JSON", "CSV", "Excel"])
                
                if st.button("Export Test Cases"):
                    if export_format == "JSON":
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(test_cases, indent=4),
                            file_name="test_cases_export.json",
                            mime="application/json"
                        )
                    elif export_format == "CSV":
                        # Flatten test cases for CSV
                        flat_cases = []
                        for tc in test_cases:
                            flat_tc = tc.copy()
                            if 'steps' in flat_tc:
                                flat_tc['steps'] = flat_tc['steps'].replace('\n', ' ')
                            if 'expected_result' in flat_tc:
                                flat_tc['expected_result'] = flat_tc['expected_result'].replace('\n', ' ')
                            if 'preconditions' in flat_tc:
                                flat_tc['preconditions'] = flat_tc['preconditions'].replace('\n', ' ')
                            flat_cases.append(flat_tc)
                        
                        csv_df = pd.DataFrame(flat_cases)
                        csv = csv_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="test_cases_export.csv",
                            mime="text/csv"
                        )
                    else:  # Excel
                        try:
                            import openpyxl
                            # Flatten test cases for Excel
                            flat_cases = []
                            for tc in test_cases:
                                flat_tc = tc.copy()
                                if 'steps' in flat_tc:
                                    flat_tc['steps'] = flat_tc['steps'].replace('\n', ' ')
                                if 'expected_result' in flat_tc:
                                    flat_tc['expected_result'] = flat_tc['expected_result'].replace('\n', ' ')
                                if 'preconditions' in flat_tc:
                                    flat_tc['preconditions'] = flat_tc['preconditions'].replace('\n', ' ')
                                flat_cases.append(flat_tc)
                            
                            excel_df = pd.DataFrame(flat_cases)
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                excel_df.to_excel(writer, index=False)
                            
                            excel_data = excel_buffer.getvalue()
                            st.download_button(
                                label="Download Excel",
                                data=excel_data,
                                file_name="test_cases_export.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except ImportError:
                            st.error("Excel export requires openpyxl. Please install it with: pip install openpyxl")
                        except Exception as e:
                            st.error(f"Excel export failed: {e}") 