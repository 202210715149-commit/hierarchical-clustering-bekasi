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

kolom = [
    "Kosong",
    "Kecamatan",
    "Negeri 2021",
    "Negeri 2022",
    "Negeri 2023",
    "Negeri 2024",
    "Jumlah Negeri",
    "Rata Negeri",
    "Swasta 2021",
    "Swasta 2022",
    "Swasta 2023",
    "Swasta 2024",
    "Jumlah Swasta",
    "Rata Swasta"
]

df_excel = pd.read_excel(
    "OLAH DATA.xlsx",
    skiprows=5,
    header=None,
    names=kolom
)

df_excel = df_excel.drop(columns=["Kosong"])

# Menghapus kolom kosong
df_excel = df_excel.dropna(axis=1, how="all")

# Menghapus baris kosong
df_excel = df_excel.dropna(how="all")

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

<div class="sidebar-footer">
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

    st.subheader("🗂️ Dataset Penelitian")

    st.info("""
**Penjelasan Dataset**

Dataset yang digunakan dalam penelitian ini merupakan data jumlah murid SMP Negeri dan SMP Swasta pada 12 kecamatan di Kota Bekasi selama periode 2021–2024. Data diperoleh dari Badan Pusat Statistik (BPS) dan Data Pokok Pendidikan (Dapodik).

Sebelum dilakukan proses Hierarchical Clustering, data terlebih dahulu melalui tahap preprocessing berupa perhitungan jumlah dan rata-rata jumlah murid SMP Negeri dan SMP Swasta pada masing-masing kecamatan. Kolom Jumlah merupakan hasil agregasi data, sedangkan kolom Rata-rata digunakan sebagai variabel input dalam proses Hierarchical Clustering.
""")

    st.caption("💡 Dataset ini merupakan data asli sebelum dilakukan proses clustering.")

    df_tampil = df_excel.copy()

    # Nomor urut mulai dari 1
    df_tampil.index = range(1, len(df_tampil) + 1)

    df_tampil = df_tampil.rename(columns={

    "Negeri 2021": "Murid Negeri 2021",
    "Negeri 2022": "Murid Negeri 2022",
    "Negeri 2023": "Murid Negeri 2023",
    "Negeri 2024": "Murid Negeri 2024",

    "Swasta 2021": "Murid Swasta 2021",
    "Swasta 2022": "Murid Swasta 2022",
    "Swasta 2023": "Murid Swasta 2023",
    "Swasta 2024": "Murid Swasta 2024",

    "Jumlah Negeri": "Jumlah Murid Negeri",
    "Jumlah Swasta": "Jumlah Murid Swasta",

    "Rata Negeri": "Rata-rata Negeri",
    "Rata Swasta": "Rata-rata Swasta"
})

    st.dataframe(
    df_tampil,
    use_container_width=True,
    height=450
)

# =====================
# Diagram Batang
# =====================

elif menu=="📊 Diagram Batang":

    st.subheader("📊 Diagram Batang")

    st.info("""
**Penjelasan Diagram Batang**

Diagram batang digunakan untuk membandingkan rata-rata jumlah murid SMP Negeri dan SMP Swasta pada setiap kecamatan di Kota Bekasi. Visualisasi ini memudahkan pengguna dalam melihat perbedaan jumlah murid antar kecamatan sehingga karakteristik setiap wilayah dapat diamati sebelum dilakukan proses clustering.
""")

    st.caption("📈 Grafik menunjukkan perbandingan rata-rata jumlah murid pada masing-masing kecamatan.")

    st.success("""
**Kesimpulan Singkat**

Grafik menunjukkan bahwa rata-rata jumlah murid pada setiap kecamatan memiliki variasi yang cukup besar. Perbedaan ini menjadi dasar dalam proses pengelompokan menggunakan Hierarchical Clustering.
""")

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15,6))

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

    st.info("""
**Penjelasan Dendrogram**

Dendrogram merupakan visualisasi hasil Hierarchical Clustering yang menunjukkan proses penggabungan antar kecamatan berdasarkan tingkat kemiripan data. Semakin rendah posisi penggabungan pada dendrogram, semakin tinggi tingkat kemiripan antar kecamatan tersebut. Pada penelitian ini dendrogram digunakan untuk membantu menentukan jumlah cluster yang akan dibentuk.
""")

    st.caption("🌳 Dendrogram membantu menunjukkan proses pembentukan 3 cluster pada penelitian ini.")

    st.success("""
**Kesimpulan Singkat**

Dendrogram memperlihatkan hubungan kemiripan antar kecamatan. Berdasarkan proses penggabungan tersebut, dipilih 3 cluster sebagai hasil akhir penelitian.
""")

    X=df[["Rata_Negeri","Rata_Swasta"]]

    linked=linkage(X,method="ward")

    fig,ax=plt.subplots(figsize=(15,6))

    ax.set_title(
    "Dendogram Hierarchical Clustering",
    fontsize=14,
    pad=15
)

    ax.set_xlabel("Kecamatan", fontsize=11)
    ax.set_ylabel("Jarak Euclidean", fontsize=11)

    # Agar layout tidak terpotong
    plt.tight_layout()

    st.pyplot(fig)

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

    st.info("""
**Penjelasan Scatter Plot**

Scatter plot digunakan untuk menampilkan hasil pengelompokan (cluster) dalam bentuk titik berdasarkan dua variabel, yaitu rata-rata jumlah murid SMP Negeri dan rata-rata jumlah murid SMP Swasta. Setiap warna menunjukkan kelompok (cluster) yang berbeda sehingga memudahkan dalam melihat persebaran data dan hubungan antar kecamatan.
""")

    st.caption("🎨 Warna yang berbeda menunjukkan kelompok (cluster) yang berbeda.")

    st.success("""
**Kesimpulan Singkat**

Scatter Plot menunjukkan bahwa setiap cluster memiliki pola persebaran yang berbeda berdasarkan rata-rata jumlah murid SMP Negeri dan SMP Swasta.
""")

    X=df[["Rata_Negeri","Rata_Swasta"]]

    model=AgglomerativeClustering(n_clusters=3)

    cluster=model.fit_predict(X)

    fig,ax=plt.subplots(figsize=(15,6))

    ax.set_title(
    "Scatter Plot Hierarchical Clustering",
    fontsize=14,
    pad=15
)

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

    st.info("""
**Penjelasan Hasil Clustering**

Halaman ini menampilkan hasil akhir proses Hierarchical Clustering berupa pembagian 12 kecamatan ke dalam 3 cluster. Setiap cluster berisi kecamatan yang memiliki karakteristik jumlah murid yang relatif mirip berdasarkan dua variabel penelitian, yaitu rata-rata jumlah murid SMP Negeri dan SMP Swasta. Hasil ini dapat digunakan sebagai dasar analisis untuk mengetahui karakteristik masing-masing kelompok kecamatan.
""")

    st.caption("🧩 Hasil clustering dapat digunakan sebagai dasar analisis karakteristik setiap kelompok kecamatan.")

    st.success("""
**Kesimpulan Singkat**

Hasil pengelompokan membagi 12 kecamatan menjadi 3 kelompok berdasarkan kemiripan karakteristik jumlah murid. Setiap cluster dapat digunakan sebagai dasar analisis lebih lanjut mengenai kondisi pendidikan di masing-masing wilayah.
""")

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

    # Menambahkan interpretasi
    interpretasi = {
    1: "Tinggi",
    2: "Rendah",
    3: "Sedang"
}

    df_cluster["Interpretasi"] = df_cluster["Cluster"].map(interpretasi)




    # ============================
    # Tabel
    # ============================

    df_tampil = df_cluster.copy()

    # Mengubah nama kolom agar lebih jelas
    df_tampil = df_tampil.rename(columns={
    "Rata_Negeri": "Rata-rata Negeri",
    "Rata_Swasta": "Rata-rata Swasta",
    "Interpretasi": "Kategori"
    })

    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(
    df_tampil,
    use_container_width=True,
    height=450
    )

    st.success("Jumlah Cluster yang terbentuk : 3")
