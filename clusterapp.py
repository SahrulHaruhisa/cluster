import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
# 1. Load Dataset
df = pd.read_csv('MALARIA_cleaned.csv')

@st.cache_resource
def load_model():
    with open('final_model_rfz.pkl', 'rb') as f:
        return pickle.load(f)

data = load_model()

# 2. Ekstrak komponen dari pickle
features = data['features']
K_range = data['K_range']
wcss = data['wcss']
scaler = data['scaler']
all_possible_risk_names = data.get('all_possible_risk_names', {})

# 3. Plot Elbow Method
st.subheader("Elbow Method Plot")
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(K_range, wcss, marker='o')
ax_elbow.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig_elbow)

# 4. Input User
n_cluster = st.number_input("Masukkan Jumlah Cluster:", 2, 10, 3)

# 5. Proses Clustering & Relabeling Otomatis
X = df[features]
X_scaled = scaler.transform(X) # Gunakan transform (bukan fit_transform) agar konsisten dengan model


kmeans = KMeans(n_clusters=n_cluster, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# 6. Visualisasi Scatter Plot
st.divider()
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("Sumbu X:", features, index=0)
with col2:
    y_axis = st.selectbox("Sumbu Y:", features, index=1 if len(features) > 1 else 0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x=x_axis,
    y=y_axis,
    hue='Cluster', 
    palette='Set1',
    s=100,
    ax=ax
)
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

raw_labels = kmeans.fit_predict(X_scaled)
df['Temp_Cluster'] = raw_labels


cluster_means = df.groupby('Temp_Cluster')['jumlah_kasus'].mean().sort_values()
mapping_new_id = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}

df['Cluster'] = df['Temp_Cluster'].map(mapping_new_id)

current_labels = all_possible_risk_names.get(n_cluster, [f"Cluster {i}" for i in range(n_cluster)])

df['Label_Keparahan'] = df['Cluster'].apply(lambda x: current_labels[x] if x < len(current_labels) else f"Level {x}")

# Hapus kolom sementara
df.drop(columns=['Temp_Cluster'], inplace=True)

# --- Menampilkan Hasil di Streamlit ---
st.subheader("Hasil Pengelompokan Data (Clustering)")

# Mengatur urutan kolom agar label muncul di depan
display_cols = ['Cluster', 'Label_Keparahan'] + [c for c in df.columns if c not in ['Cluster', 'Label_Keparahan', 'Cluster']]
st.dataframe(df[display_cols])

# --- Bagian Evaluasi Klastering (Silhouette & Davies-Bouldin) ---
# --- 6. Bagian Evaluasi Klastering (Mencegah Error k=1) ---
st.divider()
st.subheader("Evaluasi Kualitas Klastering")

# Perbaikan: Filter K_range agar hanya angka > 1 yang dihitung
k_eval = [k for k in K_range if k > 1]
silhouette_scores = []
db_scores = []

for k in k_eval:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Visualisasi Metrik
col_ev1, col_ev2 = st.columns(2)

with col_ev1:
    st.write("**Silhouette Score** (Lebih tinggi = Lebih baik)")
    fig_sil, ax_sil = plt.subplots()
    ax_sil.plot(k_eval, silhouette_scores, marker='o', color='#2ecc71')
    ax_sil.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_sil)

with col_ev2:
    st.write("**Davies-Bouldin Index** (Lebih rendah = Lebih baik)")
    fig_db, ax_db = plt.subplots()
    ax_db.plot(k_eval, db_scores, marker='o', color='#e74c3c')
    ax_db.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_db)

    # 4. Tabel Detail Angka
with st.expander("Lihat Detail Angka Evaluasi (k=2 - k=10)"):
    metrics_df = pd.DataFrame({
        'Jumlah Klaster (k)': k_eval,
        'Silhouette Score': silhouette_scores,
        'Davies-Bouldin Index': db_scores
    })
    st.dataframe(metrics_df, use_container_width=True)# 4. Tabel Detail Angka

raw_labels = kmeans.fit_predict(X_scaled)
df['Temp_Cluster'] = raw_labels
cluster_means = df.groupby('Temp_Cluster')['jumlah_kasus'].mean().sort_values()
mapping_new_id = {}
for new_label, old_label in enumerate(cluster_means.index):
    mapping_new_id[old_label] = new_label
df['Cluster'] = df['Temp_Cluster'].map(mapping_new_id)
df.drop(columns=['Temp_Cluster'], inplace=True)
score = silhouette_score(X_scaled, df['Cluster'])
print(f"\nSilhouette Score: {score:.4f}")
summary = df.groupby('Cluster')[features].mean()
print("\nRata-rata per Cluster:")
print(summary)

