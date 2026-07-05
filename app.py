import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# Memanggil CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon="📊",
    layout="wide"
)

# =====================
# Load Data
# =====================

df = pd.read_csv("dataset_clean.csv")

# =====================
# Sidebar
# =====================

#st.sidebar.markdown("## 🏠 Dashboard")

menu = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "🗂️ Dataset",
        "📊 Diagram Batang",
        "🌳 Dendrogram",
        "🎯 Scatter Plot",
        "🧩 Hasil Clustering"
    ]
)

# Mendorong tulisan ke bagian bawah sidebar
st.sidebar.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

st.sidebar.markdown("""

<div style="text-align:center; font-size:14px; color:white;">
<b>Nikmah Azizah</b><br>
Program Studi Informatika
</div>
""", unsafe_allow_html=True)

# =====================
# Dashboard
# =====================

if menu == "🏠 Dashboard":

    st.markdown("""
    <h1 style='text-align:center; color:#222222;'>
    Pengelompokan Kecamatan di Kota Bekasi Berdasarkan Jumlah Murid SMP Negeri dan Swasta Menggunakan Algoritma Hierarchical Clustering
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("🏙️ Kecamatan", "12")
    col2.metric("📊 Variabel", "2")
    col3.metric("🧩 Cluster", "3")

    st.markdown("---")

    st.subheader("📖 Tentang Penelitian")

    st.write("""
    Penelitian ini bertujuan untuk mengelompokkan 12 kecamatan di Kota Bekasi
    berdasarkan rata-rata jumlah murid SMP Negeri dan SMP Swasta menggunakan
    algoritma Hierarchical Clustering. Data yang digunakan merupakan data jumlah
    murid selama periode 2021–2024 yang diperoleh dari Badan Pusat Statistik (BPS)
    dan Data Pokok Pendidikan (Dapodik).

    Perbedaan jumlah murid antar kecamatan menyebabkan karakteristik setiap wilayah
    menjadi berbeda. Oleh karena itu, diperlukan proses pengelompokan (clustering)
    agar kecamatan yang memiliki karakteristik jumlah murid yang mirip dapat berada
    dalam satu kelompok. Hasil pengelompokan ini diharapkan dapat memberikan
    gambaran mengenai pola persebaran jumlah murid di Kota Bekasi.

    Melalui visualisasi berupa diagram batang, dendrogram, scatter plot, dan hasil
    cluster, penelitian ini dapat membantu memahami karakteristik masing-masing
    kecamatan sehingga dapat menjadi informasi pendukung dalam evaluasi pemerataan
    pendidikan serta pengambilan keputusan oleh pihak terkait.
    """)

    st.subheader("📚 Tentang Hierarchical Clustering")

    st.write("""
    Hierarchical Clustering merupakan salah satu metode dalam data mining yang
    digunakan untuk mengelompokkan data berdasarkan tingkat kemiripan antar objek.
    Metode ini membentuk struktur pengelompokan secara bertahap sehingga hubungan
    antar data dapat divisualisasikan dalam bentuk dendrogram.

    Pada penelitian ini digunakan metode Agglomerative Hierarchical Clustering,
    yaitu proses pengelompokan dimulai dari setiap data sebagai satu cluster
    kemudian digabungkan secara bertahap berdasarkan jarak terdekat menggunakan
    Ward Linkage dan Euclidean Distance hingga terbentuk tiga cluster.

    Variabel yang digunakan dalam penelitian ini terdiri dari:

    • Rata-rata jumlah murid SMP Negeri

    • Rata-rata jumlah murid SMP Swasta

    Hasil clustering memberikan informasi mengenai kecamatan yang memiliki
    karakteristik jumlah murid tinggi, sedang, maupun rendah. Informasi tersebut
    ditampilkan dalam bentuk tabel hasil clustering, diagram batang, dendrogram,
    serta scatter plot sehingga lebih mudah dipahami oleh pengguna.
    """)

    st.markdown("---")

# =====================
# Dataset
# =====================

elif menu == "🗂️ Dataset":

    st.write("🗂️ Dataset")

    df_tampil = df.copy()

    if "Cluster" in df_tampil.columns:
        df_tampil = df_tampil.drop(columns=["Cluster"])
        
    # Nomor urut mulai dari 1
    df_tampil.index = range(1, len(df_tampil) + 1)

    st.dataframe(df_tampil, use_container_width=True)

# =====================
# Diagram Batang
# =====================

elif menu=="📊 Diagram Batang":

    st.subheader("📊 Diagram Batang")

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9,6))

    ax.bar(
        x-width/2,
        df["Rata_Negeri"],
        width,
        color="#1f77b4",
        label="Negeri"
    )

    ax.bar(
        x+width/2,
        df["Rata_Swasta"],
        width,
        color="#ff7f0e",
        label="Swasta"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        df["Kecamatan"],
        rotation=35,
        ha="right"
    )

    ax.set_title("Perbandingan Rata-rata Murid Negeri dan Swasta")

    ax.set_xlabel("Kecamatan")

    ax.set_ylabel("Rata-rata Murid")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.legend()

    st.pyplot(fig)

# =====================
# Dendrogram
# =====================

elif menu=="🌳 Dendrogram":

    st.subheader("🌳 Dendrogram")

    X=df[["Rata_Negeri","Rata_Swasta"]]

    linked=linkage(X,method="ward")

    fig,ax=plt.subplots(figsize=(10,5))

    dendrogram(
        linked,
        labels=df["Kecamatan"].values,
        leaf_rotation=90
    )

    st.pyplot(fig)

# =====================
# Scatter Plot
# =====================

elif menu=="🎯 Scatter Plot":

    st.subheader("🎯 Scatter Plot")

    X=df[["Rata_Negeri","Rata_Swasta"]]

    model=AgglomerativeClustering(n_clusters=3)

    cluster=model.fit_predict(X)

    fig,ax=plt.subplots(figsize=(8,6))

    scatter=ax.scatter(
        df["Rata_Negeri"],
        df["Rata_Swasta"],
        c=cluster,
        s=150
    )

    for i in range(len(df)):
        ax.text(
            df["Rata_Negeri"][i],
            df["Rata_Swasta"][i],
            df["Kecamatan"][i]
        )

    ax.set_xlabel("Rata Negeri")

    ax.set_ylabel("Rata Swasta")

    st.pyplot(fig)

# =====================
# Hasil Clustering
# =====================

elif menu == "🧩 Hasil Clustering":

    st.subheader("🧩 Hasil Clustering")

    # Mengambil dua variabel
    X = df[["Rata_Negeri", "Rata_Swasta"]]

    # Membuat model Hierarchical Clustering
    model = AgglomerativeClustering(n_clusters=3)

    # Menentukan cluster
    cluster = model.fit_predict(X)

    # Menambahkan kolom Cluster
    df_cluster = df.copy()
    df_cluster["Cluster"] = cluster + 1

    # Mengurutkan berdasarkan cluster
    df_cluster = df_cluster.sort_values("Cluster").reset_index(drop=True)

    # Nomor urut mulai dari 1
    df_cluster.index = df_cluster.index + 1

    # Menampilkan tabel
    st.dataframe(df_cluster, use_container_width=True)

    st.success("Jumlah Cluster yang terbentuk : 3")
