import os

directory = "/home/paul/Desktop/Databases/Embeddings/EmbeddingTest8962/mpnet_output"
files = os.listdir(directory)

# Сортировка файлов по имени (по умолчанию - в алфавитном порядке)
sorted_files = sorted(files)

# Вывод отсортированного списка файлов
print(sorted_files)