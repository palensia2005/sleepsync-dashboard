
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title='SleepSync',
    page_icon='🛌',
    layout='wide'
)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    file_id = '1hEhusQq2sIv6PZQwDi2Q7NFWprBCdXz_'
    url = f'https://drive.google.com/uc?id={file_id}'
    df = pd.read_csv(url)
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    return df

df = load_data()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title('🛌 SleepSync')
st.sidebar.write('Navigasi Dashboard')
menu = st.sidebar.radio('Pilih Halaman:', [
    '🏠 Beranda',
    '📊 Distribusi Data',
    '🔍 Analisis Korelasi',
    '💡 Insight'
])

# ============================================================
# HALAMAN BERANDA
# ============================================================
if menu == '🏠 Beranda':
    st.title('🛌 SleepSync')
    st.write('Dashboard analisis kesehatan tidur berdasarkan gaya hidup')

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Data', f'{len(df)} orang')
    col2.metric('Gangguan Tidur', '2 jenis')
    col3.metric('Fitur Analisis', '12 kolom')

# ============================================================
# HALAMAN DISTRIBUSI DATA
# ============================================================
elif menu == '📊 Distribusi Data':
    st.title('📊 Distribusi Data')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Distribusi Gangguan Tidur')
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Sleep Disorder', palette='Set2', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader('Distribusi Kualitas Tidur')
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x='Quality of Sleep', bins=10, kde=True, ax=ax2)
        st.pyplot(fig2)

    st.subheader('Durasi Tidur per Pekerjaan')
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=df, x='Occupation', y='Sleep Duration', palette='Set3', ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

# ============================================================
# HALAMAN ANALISIS KORELASI
# ============================================================
elif menu == '🔍 Analisis Korelasi':
    st.title('🔍 Analisis Korelasi')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Stress Level vs Kualitas Tidur')
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Stress Level', y='Quality of Sleep', palette='Reds', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader('Durasi Tidur vs Kualitas Tidur')
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='Sleep Duration', y='Quality of Sleep',
                       hue='Sleep Disorder', palette='Set2', ax=ax2)
        st.pyplot(fig2)

    st.subheader('Heatmap Korelasi')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(),
               annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# ============================================================
# HALAMAN INSIGHT
# ============================================================
elif menu == '💡 Insight':
    st.title('💡 Insight')

    st.info('**Stress Level & Kualitas Tidur**\n\nSemakin tinggi tingkat stres, semakin rendah kualitas tidur.')
    st.info('**Durasi Tidur & Kualitas Tidur**\n\nDurasi tidur 7-8 jam berkorelasi dengan kualitas tidur terbaik.')
    st.info('**Gangguan Tidur**\n\nPenderita Insomnia memiliki tingkat aktivitas fisik paling rendah.')
