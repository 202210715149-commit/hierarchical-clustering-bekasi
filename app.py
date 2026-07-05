import streamlit as st
import pandas as pd

# Memanggil CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Hierarchical Clustering")
st.write("### Pengelompokan 12 Kecamatan Berdasarkan Sekolah Negeri dan Swasta")

# =====================
# Load Data
# =====================

df = pd.read_csv("dataset_clean.csv")

# =====================
# Sidebar
# =====================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Dataset",
        "Diagram Batang",
        "Dendrogram",
        "Scatter Plot",
        "Hasil Cluster"
    ]
)

# =====================
# Dataset
# =====================

if menu=="Dataset":

    st.subheader("Dataset")

    st.dataframe(df,use_container_width=True)

# =====================
# Diagram Batang
# =====================

elif menu=="Diagram Batang":

    st.subheader("Diagram Batang")

    fig, ax = plt.subplots(figsize=(10,5))

    x = range(len(df))

    ax.bar(x,df["Rata_Negeri"],label="Negeri")

    ax.bar(x,df["Rata_Swasta"],
           bottom=df["Rata_Negeri"],
           label="Swasta")

    ax.set_xticks(x)

    ax.set_xticklabels(df["Kecamatan"],rotation=60)

    ax.legend()

    st.pyplot(fig)

# =====================
# Dendrogram
# =====================

elif menu=="Dendrogram":

    st.subheader("Dendrogram")

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

elif menu=="Scatter Plot":

    st.subheader("Scatter Plot")

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

elif menu=="Hasil Cluster":

    st.subheader("Hasil Clustering")

    X=df[["Rata_Negeri","Rata_Swasta"]]

    model=AgglomerativeClustering(n_clusters=3)

    df["Cluster"]=model.fit_predict(X)+1

    st.dataframe(df,use_container_width=True)

    st.success("Jumlah Cluster : 3")