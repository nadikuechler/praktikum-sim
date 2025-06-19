import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv", encoding='latin1')
    return df

df = load_data()

# Cleaning Data
cols_to_drop = [f'Unnamed: {i}' for i in range(12, 26)]
df = df.drop(columns=cols_to_drop)
df.dropna(subset=['director', 'cast', 'country', 'rating', 'duration'], inplace=True)
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Dataset", "EDA", "Visualisasi", "RFM"])

# Page: Dataset
if page == "Dataset":
    st.title("Dataset Netflix")
    st.write("Data setelah dibersihkan:")
    st.dataframe(df)
    st.write("Jumlah data:", len(df))

# Page: EDA
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Proporsi Movie vs TV Show")
    st.write(df['type'].value_counts())

    st.subheader("10 Negara Asal Konten Terbanyak")
    st.write(df['country'].value_counts().head(10))

    st.subheader("10 Genre Paling Sering Muncul")
    genre_series = df['listed_in'].dropna().apply(lambda x: [i.strip() for i in x.split(',')])
    all_genres = Counter([genre for sublist in genre_series for genre in sublist])
    st.write(pd.Series(dict(all_genres)).sort_values(ascending=False).head(10))

    st.subheader("10 Sutradara Paling Produktif")
    st.write(df['director'].dropna().value_counts().head(10))

    st.subheader("10 Aktor/Aktris Paling Sering Muncul")
    cast_series = df['cast'].dropna().apply(lambda x: [i.strip() for i in x.split(',')])
    all_casts = Counter([actor for sublist in cast_series for actor in sublist])
    st.write(pd.Series(dict(all_casts)).sort_values(ascending=False).head(10))

    st.subheader("Jumlah Konten Rilis per Tahun (10 Terakhir)")
    st.write(df['release_year'].value_counts().sort_index(ascending=False).head(10))

    st.subheader("Konten Lama (rilis < 2000) yang Baru Ditambahkan")
    konten_lama = df[(df['release_year'] < 2000) & (df['year_added'].notna())][['title', 'release_year', 'date_added']]
    st.dataframe(konten_lama.head())


# Page: Visualisasi
elif page == "Visualisasi":
    st.title("Visualisasi dan Analisis Penjelas")

    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    st.subheader("Distribusi Movie vs TV Show")
    type_counts = df['type'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
    ax1.set_title("Distribusi Tipe Tayangan (Movie vs TV Show)")
    ax1.axis('equal')
    st.pyplot(fig1)

    genre_series = df['listed_in'].dropna().apply(lambda x: [i.strip() for i in x.split(',')])
    all_genres = Counter([genre for sublist in genre_series for genre in sublist])
    
    st.subheader("Top 10 Genre Paling Sering Muncul")
    fig2, ax2 = plt.subplots()
    pd.Series(dict(all_genres)).sort_values(ascending=False).head(10).plot(kind='barh', color='lightcoral', ax=ax2)
    ax2.set_title('10 Genre Paling Sering Muncul')
    ax2.set_xlabel('Jumlah Tayangan')
    ax2.invert_yaxis()
    st.pyplot(fig2)

    st.subheader("10 Negara Asal Teratas")
    fig3, ax3 = plt.subplots()
    df['country'].value_counts().head(10).plot(kind='bar', color='mediumseagreen', ax=ax3)
    ax3.set_title('10 Negara Asal Teratas')
    ax3.set_xlabel('Negara')
    ax3.set_ylabel('Jumlah Tayangan')
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.subheader("10 Sutradara Terproduktif")
    fig4, ax4 = plt.subplots()
    df['director'].value_counts().head(10).plot(kind='barh', color='mediumpurple', ax=ax4)
    ax4.set_title('10 Sutradara Terproduktif')
    ax4.set_xlabel('Jumlah Tayangan')
    ax4.invert_yaxis()
    st.pyplot(fig4)

    cast_series = df['cast'].dropna().apply(lambda x: [i.strip() for i in x.split(',')])
    all_casts = Counter([actor for sublist in cast_series for actor in sublist])

    st.subheader("10 Aktor/Aktris Terproduktif")
    fig5, ax5 = plt.subplots()
    pd.Series(dict(all_casts)).sort_values(ascending=False).head(10).plot(kind='barh', color='orange', ax=ax5)
    ax5.set_title('10 Aktor/Aktris Terproduktif')
    ax5.set_xlabel('Jumlah Penampilan')
    ax5.invert_yaxis()
    st.pyplot(fig5)

    st.subheader("Tahun Rilis dengan Konten Terbanyak")
    fig6, ax6 = plt.subplots()
    df['release_year'].value_counts().sort_values(ascending=False).head(15).plot(kind='bar', color='steelblue', ax=ax6)
    ax6.set_title('Tahun Rilis dengan Konten Terbanyak')
    ax6.set_xlabel('Tahun Rilis')
    ax6.set_ylabel('Jumlah Konten')
    ax6.tick_params(axis='x', rotation=45)
    st.pyplot(fig6)

    st.subheader("Jumlah Tayangan berdasarkan Tahun Rilis dan Rating")
    heatmap_data = df.pivot_table(index='release_year', columns='rating', aggfunc='size', fill_value=0)
    heatmap_data = heatmap_data[heatmap_data.index >= 2000]
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', ax=ax7)
    ax7.set_title('Jumlah Tayangan per Tahun Rilis dan Rating')
    ax7.set_xlabel('Rating')
    ax7.set_ylabel('Tahun Rilis')
    st.pyplot(fig7)

    st.subheader("Jumlah Tayangan Ditambahkan ke Netflix per Tahun")
    yearly_counts = df['year_added'].dropna().astype(int).value_counts().sort_index()
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    ax8.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-', color='teal')
    ax8.set_title('Jumlah Tayangan Ditambahkan per Tahun')
    ax8.set_xlabel('Tahun')
    ax8.set_ylabel('Jumlah Tayangan')
    ax8.grid(True)
    st.pyplot(fig8)

    st.subheader("Segmentasi Genre Populer Berdasarkan Tipe Tayangan")
    df['main_genre'] = df['listed_in'].dropna().apply(lambda x: x.split(',')[0].strip())
    segment3 = df.groupby(['type', 'main_genre']).size().unstack().fillna(0)
    segment3 = segment3.loc[:, segment3.sum().sort_values(ascending=False).head(10).index]
    fig9, ax9 = plt.subplots(figsize=(12,6))
    segment3.T.plot(kind='bar', ax=ax9)
    ax9.set_title('Segmentasi Genre Populer Berdasarkan Tipe Tayangan')
    ax9.set_xlabel('Genre')
    ax9.set_ylabel('Jumlah Tayangan')
    ax9.tick_params(axis='x', rotation=45)
    st.pyplot(fig9)

    st.subheader("Segmentasi Tipe Tayangan Berdasarkan Tahun Rilis")
    segment4 = df.groupby(['release_year', 'type']).size().unstack().fillna(0).tail(15)
    fig10, ax10 = plt.subplots(figsize=(12,6))
    segment4.plot(kind='line', marker='o', ax=ax10)
    ax10.set_title('Segmentasi Tipe Tayangan Berdasarkan Tahun Rilis')
    ax10.set_xlabel('Tahun Rilis')
    ax10.set_ylabel('Jumlah Tayangan')
    ax10.grid(True)
    st.pyplot(fig10)

# Page: RFM Analysis (berdasarkan 'date_added' sebagai proxy 'recency')
elif page == "RFM":
    st.title("RFM Analysis")

    st.subheader("Negara Asal Konten Terbanyak yang Diunggah (1 Tahun Terakhir)")

    recent = df[df['year_added'] >= (df['year_added'].max() - 1)]
    recent_genres = recent['listed_in'].dropna().str.split(', ')
    all_recent_genres = pd.Series([g for sub in recent_genres for g in sub])
    top_genres = all_recent_genres.value_counts().head(10)

    countries = df['country'].dropna().str.split(', ')
    all_countries = pd.Series([c.strip() for sub in countries for c in sub])
    top_countries = all_countries.value_counts().head(10)

    df_countries = pd.DataFrame({
        'Negara': top_countries.index,
        'Jumlah': top_countries.values
    })

    fig, ax = plt.subplots()
    sns.barplot(data=df_countries, x='Jumlah', y='Negara', hue='Negara', dodge=False, palette='viridis', ax=ax)
    ax.set_title("Top 10 Negara Asal Konten Netflix")
    ax.legend().remove()
    st.pyplot(fig)