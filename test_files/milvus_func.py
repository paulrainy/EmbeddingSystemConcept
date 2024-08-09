from pymilvus import MilvusClient, DataType

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530


def connect_to_milvus():
    client = MilvusClient(
        uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    )
    return client


def create_collection(client, collection_name):
    if client.has_collection(collection_name):

        print(f"Коллекция '{collection_name}' уже существует.")

    else:

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="idx", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="idx",
            index_type="STL_SORT"
        )

        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        print(f"Коллекция '{collection_name}' создана.")


def drop_collection(client, collection_name):
    client.drop_collection(
        collection_name=collection_name
    )
    print(f"Коллекция '{collection_name}' сброшена.")


# А вот тут вот вопрос к тебе. Где будет происходить предобработка data? В этой функции или в main?
def input_to_collection(client, collection_name, data):
    client.insert(
        collection_name=collection_name,
        data=data
    )


def vector_search(client, collection_name, vector):
    res = client.search(
        collection_name=collection_name,
        anns_field="vector",
        data=vector,
        limit=5,
        search_params={"metric_type": "L2", "params": {}},  # Search parameters
    )
    return res


# Функция для получения количества записей в коллекции
def data_count(client, collection_name):
    stats = client.get_collection_stats(collection_name)
    return stats['row_count']


# Функция для получения записи по id
def get_data_by_id(client, collection_name, id):
    res = client.get(
        collection_name=collection_name,
        ids=[id]
    )
    return res


# Функция для получения всех данных
# Эту функцию я мб переделаю, но для прототипа она вроде как норм
def get_all_data(client, collection_name):
    stats = client.get_collection_stats(collection_name)
    #print(stats)
    res = client.query(
        collection_name=collection_name,
        limit=stats['row_count']
    )
    return res
