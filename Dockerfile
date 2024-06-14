# Gunakan image dasar Python
FROM python:3.9-slim

# Set lingkungan kerja di dalam container
WORKDIR /app

# Salin file requirements.txt dan install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Download dataset NLTK yang diperlukan
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

# Expose port untuk aplikasi Flask
EXPOSE 8080

# Perintah untuk menjalankan aplikasi
CMD ["python", "main.py"]
