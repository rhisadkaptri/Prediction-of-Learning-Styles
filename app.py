import streamlit as st
import joblib  # Digunakan untuk memuat model yang sudah disimpan

# Load the models
model_processing = joblib.load('model_processing.pkl')
model_perception = joblib.load('model_perception.pkl')
model_input = joblib.load('model_input.pkl')
model_understanding = joblib.load('model_understanding.pkl')

# Fungsi untuk prediksi
def predict_all_models(input_data):
    processing_pred = model_processing.predict([input_data[:11]])[0]
    perception_pred = model_perception.predict([input_data[11:22]])[0]
    input_pred = model_input.predict([input_data[22:33]])[0]
    understanding_pred = model_understanding.predict([input_data[33:44]])[0]
    return processing_pred, perception_pred, input_pred, understanding_pred

# App title
st.title("Klasifikasi Pembelajaran")

# Input email dan nama
email = st.text_input("Masukkan email Anda (harus mengandung '@binus.ac.id'):")
name = st.text_input("Masukkan nama Anda:")

# Validasi email
if '@binus.ac.id' in email and name:
    st.write("Email valid. Silakan isi form di bawah ini.")

    # Daftar pertanyaan dan jawaban
    questions = [
        "Saya memahami suatu konsep dengan lebih baik setelah saya...",
        "Ketika saya belajar sesuatu yang baru, hal ini dapat membantu saya untuk...",
        "Pada sebuah studi kelompok kerja pada materi yang sulit, saya cenderung untuk...",
        "Jika saya berada di kelas...",
        "Ketika saya menyelesaikan tugas/PR, saya lebih cenderung untuk...",
        "Saya lebih suka belajar...",
        "Pertama-tama saya lebih suka terlebih dahulu...",
        "Saya lebih mudah mengingat...",
        "Ketika saya harus mengerjakan suatu tugas kelompok, pertama kali saya ingin...",
        "Saya cenderung dianggap...",
        "Ide untuk mengerjakan pekerjaan rumah (PR) dalam kelompok, dengan satu kelas untuk seluruh kelompok,...",
        "Saya lebih suka dianggap...",
        "Jika saya menjadi seorang dosen, saya lebih suka mengajar kuliah...",
        "Saya merasa lebih mudah...",
        "Dalam membaca non-fiksi, saya lebih suka...",
        "Saya lebih suka ide dari...",
        "Saya cenderung dianggap...",
        "Ketika membaca yang saya suka, saya suka penulis yang...",
        "Ketika saya menampilkan sesuatu, saya lebih memilih untuk...",
        "Untuk menghormati seseorang, saya memanggilnya dengan...",
        "Untuk menghibur diri, saya lebih suka...",
        "Ketika saya melakukan kalkulasi (perhitungan) yang panjang...",
        "Ketika saya berpikir tentang apa yang saya lakukan kemarin, kemungkinan besar yang saya dapatkan...",
        "Saya lebih memilih untuk mendapatkan informasi baru dalam...",
        "Pada sebuah buku yang di dalamnya banyak gambar dan grafik, saya cenderung untuk...",
        "Saya suka terhadap dosen...",
        "Saya selalu mengingat...",
        "Ketika saya mendapatkan petunjuk ke sebuah tempat baru, saya lebih suka...",
        "Ketika saya melihat diagram atau sketsa di kelas, saya cenderung untuk mengingat...",
        "Ketika seseorang menunjukkan sebuah data, saya lebih suka...",
        "Ketika saya bertemu orang-orang di sebuah pesta, saya cenderung untuk mengingat...",
        "Saya lebih suka mata pelajaran yang mengutamakan...",
        "Saya cenderung menggambar tempat-tempat yang saya lihat...",
        "Saya cenderung...",
        "Setelah saya memahami...",
        "Ketika saya memecahkan masalah matematika...",
        "Ketika saya menganalisis cerita atau novel...",
        "Perlu bagi saya bahwa pengajar...",
        "Saya belajar...",
        "Ketika mempertimbangkan kumpulan informasi, saya lebih cenderung untuk...",
        "Ketika menulis sebuah makalah, saya lebih cenderung untuk...",
        "Ketika saya mempelajari suatu materi baru, saya lebih suka untuk...",
        "Pada awal perkuliahan beberapa dosen memberikan sebuah outline mengenai materi kuliah. Outline tersebut...",
        "Ketika memecahkan masalah di dalam sebuah kelompok, saya akan cenderung untuk..."
    ]

    answers = [
        ("Mencobanya", "Berpikir"),
        ("Berbicara tentang sesuatu itu", "Berpikir tentang sesuatu itu"),
        ("Bergabung dan menyumbangkan ide-ide", "Duduk di belakang dan mendengarkan"),
        ("Saya biasanya akan mengenali banyak siswa", "Saya jarang mengenali banyak siswa"),
        ("Langsung bekerja pada solusi", "Mencoba untuk memahami masalah terlebih dahulu"),
        ("Dalam kelompok", "Sendiri"),
        ("Mencoba segala sesuatu", "Berpikir tentang bagaimana saya akan melakukannya"),
        ("Sesuatu yang telah saya lakukan", "Sesuatu yang telah saya pikirkan"),
        ("Melakukan diskusi secara bersama dalam kelompok dimana setiap orang berkontribusi memberikan ide-ide", "Melakukan diskusi secara individu kemudian bersama-sama membandingkan ide-ide dalam sebuah kelompok"),
        ("Ramah", "Pendiam"),
        ("Menarik bagi saya", "Tidak menarik bagi saya"),
        ("Realistis", "Inovatif"),
        ("Yang berhubungan dengan fakta dan situasi kehidupan nyata", "Yang berhubungan dengan ide-ide dan teori"),
        ("Untuk mempelajari fakta-fakta", "Untuk mempelajari konsep"),
        ("Sesuatu yang mengajarkan saya fakta-fakta baru atau memberitahu saya bagaimana melakukan sesuatu", "Sesuatu yang memberi saya ide-ide baru untuk berpikir"),
        ("Apa yang dapat dipercaya", "Teori"),
        ("Berhati-hati dalam mengerjakan tugas secara detail", "Kreatif dalam bekerja"),
        ("Jelas mengatakan apa yang mereka maksud", "Mengatakan hal-hal yang kreatif, dengan cara yang menarik"),
        ("Menguasai salah satu cara untuk melakukannya", "Datang dengan cara-cara baru untuk melakukannya"),
        ("Bijaksana", "Imajinatif"),
        ("Menonton televisi", "Membaca buku"),
        ("Saya cenderung untuk mengulang semua langkah dan memeriksa pekerjaan saya dengan hati-hati", "Saya merasa bosan memeriksa pekerjaan saya dan harus memaksakan diri untuk melakukannya"),
        ("Sebuah gambar", "Kata-kata"),
        ("Gambar, diagram, grafik, atau peta", "Petunjuk tertulis atau informasi verbal"),
        ("Melihat pada gambar dan grafik secara hati-hati", "Fokus pada teks yang tertulis"),
        ("Yang menempatkan banyak diagram di papan", "Yang menghabiskan banyak waktu untuk menjelaskan"),
        ("Apa yang saya lihat", "Apa yang saya dengar"),
        ("Peta", "Instruksi tertulis"),
        ("Gambar", "Apa yang pengajar jelaskan tentang hal itu"),
        ("Diagram atau grafik", "Teks hasil meringkas"),
        ("Bagaimana penampilan mereka", "Apa yang mereka katakan tentang diri mereka sendiri"),
        ("Materi kongkrit (fakta, data)", "Materi abstrak (konsep, teori)"),
        ("Dengan mudah dan cukup akurat", "Dengan kesulitan dan tanpa banyak detail"),
        ("Memahami dengan detail sebuah subjek tetapi mungkin tidak terlalu jelas pada struktur keseluruhan", "Memahami struktur secara keseluruhan, tetapi mungkin tidak detail"),
        ("Semua bagian, saya memahami seluruhnya", "Secara keseluruhan, saya melihat kesesuaian dari sebab bagian-bagiannya"),
        ("Biasanya saya bekerja dengan cara saya sendiri sebagai langkah untuk menemukan solusi pada suatu waktu", "Saya sering melihat solusi tetapi kemudian harus berjuang untuk mencari tahu langkah untuk mendapatkan solusi tersebut"),
        ("Saya pikir insiden dan mencoba untuk melihat keterkaitan untuk mengetahui tema cerita tersebut", "Saya hanya tahu apa tema yang ketika saya selesai membaca dan kemudian saya harus kembali dan menemukan kejadian-kejadian yang merujuk pada tema tersebut"),
        ("Menyusun materi dalam langkah-langkah yang jelas dan berurutan", "Memberikan gambaran secara keseluruhan yang berhubungan dengan materi mata pelajaran lainnya"),
        ("Dengan konsisten. Jika saya belajar dengan giat, maka saya akan 'mengerti'", "Dalam komunitas yang tepat. Pada awalnya saya kebingungan, namun kemudian saya akan 'memahami'"),
        ("Fokus pada detail dan mengesampingkan gambaran besar", "Mencoba untuk memahami gambaran besar sebelum masuk ke detail"),
        ("Bekerja pada (berpikir tentang atau menulis) pada permulaan kertas dan terus maju ke depan", "Bekerja pada (berpikir tentang atau menulis) pada bagian kertas yang berbeda dan kemudian menyusunnya"),
        ("Tetap fokus pada hal itu, belajar materi itu sebanyak yang saya bisa", "Mencoba untuk membuat hubungan antara materi tersebut dengan materi lain yang terkait"),
        ("Sedikit membantu saya", "Sangat membantu saya"),
        ("Memikirkan langkah-langkah proses di dalam solusi", "Memikirkan konsekuensi yang mungkin terjadi atau mencari berbagai macam solusi untuk menyelesaikannya secara langsung")
    ]

    # Input data form
    with st.form("prediction_form"):
        input_data = []
        for i, question in enumerate(questions):
            answer = st.radio(question, options=answers[i], index=0)
            input_value = 1 if answer == answers[i][0] else -1
            input_data.append(input_value)
        
        # Submit button
        submitted = st.form_submit_button("Submit")

    # Jika form telah disubmit, lakukan prediksi
    if submitted:
        processing_pred, perception_pred, input_pred, understanding_pred = predict_all_models(input_data)
        
        # Tampilkan hasil prediksi
        st.markdown(f"Gaya Belajar Processing anda adalah: **{processing_pred}**")
        st.markdown(f"Gaya Belajar Perception anda adalah: **{perception_pred}**")
        st.markdown(f"Gaya Belajar Input anda adalah: **{input_pred}**")
        st.markdown(f"Gaya Belajar Understanding anda adalah: **{understanding_pred}**")
else:
    st.write("Silakan masukkan email valid dari Binus.")
