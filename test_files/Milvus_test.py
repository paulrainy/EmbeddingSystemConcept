import numpy as np
import pandas as pd
import os
from pymilvus import MilvusClient, DataType
import datetime
import json

"""
array = []
array.append("Во время обучения модели machine learning, важно обеспечить достаточное количество данных для тренировки") 
array.append("Во время обучения модели машинного обучения, важно обеспечить достаточное количество данных для тренировки") 
array.append("Использование API позволяет интегрировать различные сервисы в ваше приложение, упрощая процесс разработки") 
array.append("The use of an API allows you to integrate various services into your application, simplifying the development process")
array.append("Для анализа больших объемов данных рекомендуется использовать фреймворк Hadoop, который обеспечивает эффективную обработку данных") 
array.append("В процессе дебаггинга программы была обнаружена ошибка, связанная с неправильным использованием переменных") 
array.append("При разработке user interface необходимо учитывать принципы user experience, чтобы сделать приложение максимально удобным для пользователя") 
array.append("При разработке пользовательского интерфейса необходимо учитывать принципы пользовательского опыта, чтобы сделать приложение максимально удобным для пользователя")
"""

client = MilvusClient(
    uri="http://localhost:19530"
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(field_name="idx", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=350)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="idx",
    index_type="STL_SORT"
)

index_params.add_index(
    field_name="vector", 
    index_type="IVF_FLAT",
    metric_type="L2",
    params={ "nlist": 1 }
)

client.drop_collection(
    collection_name="test_col"
)

client.create_collection(
    collection_name="test_col",
    schema=schema,
    index_params=index_params
)

directory = "/home/paul/Desktop/Databases/Embeddings/EmbeddingTest8962/mpnet_output"
files = os.listdir(directory)

start = datetime.datetime.now()

for idx, file in enumerate(files):
    df = pd.read_csv(directory+'/'+file).to_numpy().tolist()

    data = [
        {"idx": idx, "vector": df[0], "text": file}
    ]

    client.insert(
        collection_name="test_col",
        data=data
    )

finish = datetime.datetime.now()

print (f"Time: {finish-start}")

vector_check = pd.read_csv(directory+'/mpnet_vectorized_output_456.csv').to_numpy().tolist()

start = datetime.datetime.now()

res = client.search(
    collection_name="test_col",
    #anns_field="vector",
    data=[vector_check[0]],
    limit=5, 
    search_params={"metric_type": "L2", "params": {}}, # Search parameters
    output_fields=["text"]
)

finish = datetime.datetime.now()

print (f"Time: {finish-start}")

result = json.dumps(res, indent=4)
print(result)

"""df = pd.read_csv(directory+'/sbert_vectorized_output_36.csv').to_numpy().tolist()

data = [
    {"idx": 8963, "vector": df[0]}#, "text": array[idx]}
]

start = datetime.datetime.now()

client.insert(
    collection_name="test_col",
    data=data
)

finish = datetime.datetime.now()

print (f"Time: {finish-start}")
"""
