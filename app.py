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
    <h1 style='text-align:center; color:#1E3A8A;'>
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
    berdasarkan rata-rata jumlah murid sekolah negeri dan sekolah swasta
    menggunakan metode Hierarchical Clustering.
    """)

    st.subheader("📚 Tentang Hierarchical Clustering")

    st.write("""
    Hierarchical Clustering adalah metode klaster yang dilakukan secara bertahap (hirarki) dengan menggunakan konsep penggabungan dua klaster kecil yang memiliki jarak terdekat menjadi satu klaster yang lebih besar yang disebut dengan metode Agglomeratif Hierarchical Clustering atau pemecahan klaster besar ke beberapa klaster yang lebih kecil dengan dasar ketidakmiripan paling tinggi klaster-klaster lebih kecil yang disebut dengan metode Divisive. Pembentukan anggota klaster pada metode hirarki menggunakan bagan atau dendrogram

    Pada penelitian ini digunakan metode Agglomerative
    Hierarchical Clustering dengan Ward Linkage untuk
    menghasilkan 3 cluster berdasarkan dua variabel yaitu:

    • Rata-rata Murid Sekolah Negeri

    • Rata-rata Murid Sekolah Swasta
    """)

    st.markdown("---")

# =====================
# Dataset
# =====================

elif menu=="📂 Dataset":

    st.subheader("📂 Dataset")

    st.dataframe(df,use_container_width=True)

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
# Hasil Cluster
# =====================

elif menu=="🧩 Hasil Cluster":

    st.subheader("🧩 Hasil Clustering")

    X=df[["Rata_Negeri","Rata_Swasta"]]

    model=AgglomerativeClustering(n_clusters=3)

    df["Cluster"]=model.fit_predict(X)+1

    st.dataframe(df,use_container_width=True)

    st.success("Jumlah Cluster : 3")
