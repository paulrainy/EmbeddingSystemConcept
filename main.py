import pandas as pd
import numpy as np
import re
import math
import torch
import torch.nn.functional as F
from pandas import DataFrame
from torch import Tensor
from typing import List
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient, DataType
from sklearn.cluster import KMeans


class EmbeddingPipeline:
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530

    file_path = None
    df = None
    prepared_df = None
    local_processed_data = None

    milvus_client = None
    milvus_collection_name = None
    milvus_processed_data = None

    clustered_labels = None
    clustered_centers = None

    def __init__(self, file_path: str) -> None:  # конструктор
        self.file_path = file_path

    def get_file_path(self) -> str:  # геттер пути файла
        return self.file_path

    def get_df(self) -> DataFrame:
        return self.df

    def get_prepared_df(self) -> List[DataFrame]:
        return self.prepared_df

    def get_local_processed_data(self) -> List[Tensor]:
        return self.local_processed_data

    def get_milvus_client(self) -> MilvusClient:
        return self.milvus_client

    def get_milvus_collection_name(self) -> str:
        return self.milvus_collection_name

    def get_milvus_processed_data(self) -> List[dict]:
        return self.milvus_processed_data

    def import_df(self) -> None:
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Successfully imported data from excel file {self.file_path}")
        except Exception as e:
            raise ValueError(f"Error importing file: {e}")

    def prepare_df(self) -> None:
        print("Preparing data...")

        def remove_columns_step(dataset: DataFrame) -> DataFrame:
            output_dataset = dataset[
                ["Id", "Direction", "Section", "TestCaseName", "Preconditions", "Steps", "Postconditions",
                 "ExpectedResult"]].copy()
            output_dataset[["Id", "Direction", "Section", "TestCaseName"]] = output_dataset[
                ["Id", "Direction", "Section", "TestCaseName"]].ffill()
            return output_dataset

        output_first_step = remove_columns_step(self.df)
        print("Useless columns removed!")

        def parse_tests_by_id(dataset: DataFrame) -> List[DataFrame]:
            test_cases = []
            for group_name, frame in dataset.groupby('Id'):
                test_cases.append(frame)

            return test_cases

        output_second_step = parse_tests_by_id(output_first_step)
        print("Test cases parsed!")

        def remove_empty_cells(array: List[DataFrame]) -> List[DataFrame]:
            def up_cells(tst_case: DataFrame, column: str) -> DataFrame:
                for i in range(len(tst_case[column])):
                    if i == len(tst_case.index) - 1:
                        tst_case.at[tst_case.index[i], column] = math.nan
                        break
                    else:
                        tst_case.at[tst_case.index[i], column] = tst_case.at[tst_case.index[i + 1], column]

                return tst_case

            for dframe in array:
                for col in ["Preconditions", "Steps", "Postconditions", "ExpectedResult"]:
                    dframe = up_cells(dframe, col)

                dframe["Steps"] = dframe["Steps"].fillna(dframe["Preconditions"])
                dframe["Steps"] = dframe["Steps"].fillna(dframe["Postconditions"])
                dframe.drop(["Preconditions", "Postconditions"], inplace=True, axis=1)

            return array

        output_third_step = remove_empty_cells(output_second_step)
        print("Empty cells removed!")
        print("Preparing done!")
        self.prepared_df = output_third_step

    def process_df(self) -> None:
        print("Processing data...")

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        print(f"Device is {device}")

        def remove_ids(dataset: List[DataFrame]) -> List[DataFrame]:
            array = []
            for dframe in dataset:
                array.append(dframe[["Direction", "Section", "TestCaseName", "Steps", "ExpectedResult"]])

            return array

        prepared_df_without_id = remove_ids(self.prepared_df)

        def mean_pooling(model_output, attention_mask) -> Tensor:
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        mpnet_vectorized_output_array = []

        model.to(device)

        for test_case in prepared_df_without_id:
            # Tokenize
            encoded_input = tokenizer(re.sub(r'\n+', '', str(test_case)),
                                      return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling. In this case, average pooling
            embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            mpnet_vectorized_output_array.append(F.normalize(embedding, p=2, dim=1))

        self.local_processed_data = mpnet_vectorized_output_array
        print("Processing done")

    def connect_to_milvus(self) -> None:
        print("Connecting to Milvus")

        try:
            client = MilvusClient(
                uri=f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"
            )
            self.milvus_client = client
            print(f"Connected to {self.MILVUS_HOST}:{self.MILVUS_PORT} with {self.milvus_client}")

        except ConnectionError:
            print("Failed to connect to Milvus: connection error.")
        except TimeoutError:
            print("Failed to connect to Milvus: timeout.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def set_milvus_collection_name(self, collection_name: str) -> None:
        self.milvus_collection_name = collection_name

    def create_milvus_db(self) -> None:
        if self.milvus_collection_name is None:
            raise ValueError("Milvus collection name must be specified!")

        elif self.milvus_client.has_collection(self.milvus_collection_name):
            print(f"Collection '{self.milvus_collection_name}' already exists!")

        else:
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )

            # id for milvus
            schema.add_field(field_name="idx", datatype=DataType.INT64, is_primary=True)
            # vectors
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
            # test_case ids
            schema.add_field(field_name="inner_id", datatype=DataType.INT64, is_primary=False)
            # direction of test_case
            schema.add_field(field_name="direction_name", datatype=DataType.VARCHAR, max_length=512)
            # section of test case
            schema.add_field(field_name="section_name", datatype=DataType.VARCHAR, max_length=512)
            # name of test case
            schema.add_field(field_name="test_case_name", datatype=DataType.VARCHAR, max_length=512)
            # field for steps
            schema.add_field(field_name="steps", datatype=DataType.VARCHAR, max_length=8192)
            # field for expected results
            schema.add_field(field_name="expected_result", datatype=DataType.VARCHAR, max_length=8192)

            index_params = self.milvus_client.prepare_index_params()

            index_params.add_index(
                field_name="idx",
                index_type="STL_SORT"
            )

            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128}
            )

            index_params.add_index(
                field_name="inner_id",
                index_type="STL_SORT"
            )

            self.milvus_client.create_collection(
                collection_name=self.milvus_collection_name,
                schema=schema,
                index_params=index_params
            )

            print(f"Collection '{self.milvus_collection_name}' created successfully!")

    def drop_milvus_db(self) -> None:
        try:
            self.milvus_client.drop_collection(
                collection_name=self.milvus_collection_name
            )
            print(f"Collection '{self.milvus_collection_name}' dropped successfully!")
        except Exception as e:
            print(f"An error occurred while dropping the collection '{self.milvus_collection_name}': {e}")

    def send_data_to_milvus(self) -> None:
        print("Sending data to Milvus")
        np_input_list = []
        for t_outer in self.local_processed_data:
            for t_inner in t_outer:
                np_input_list.append(t_inner.cpu().numpy())

        print(f"{len(np_input_list)}")

        for i, vector in enumerate(np_input_list):
            milvus_output_list = [{"idx": i,
                                   "vector": vector,
                                   "inner_id": int(self.prepared_df[i].at[self.prepared_df[i].index[0], "Id"]),
                                   "direction_name": str(self.prepared_df[i].at[
                                       self.prepared_df[i].index[0], "Direction"]),
                                   "section_name": str(self.prepared_df[i].at[
                                       self.prepared_df[i].index[0], "Section"]),
                                   "test_case_name": str(self.prepared_df[i].at[
                                       self.prepared_df[i].index[0], "TestCaseName"]),
                                   "steps": re.sub(r'\n+', '', str(self.prepared_df[i]["Steps"])),
                                   "expected_result": re.sub(r'\n+', '', str(self.prepared_df[i]["ExpectedResult"])),
                                   }]
            self.milvus_client.insert(
                collection_name=self.milvus_collection_name,
                data=milvus_output_list
            )
            milvus_output_list.clear()

        print("Data sent to Milvus!")

    def milvus_data_count(self):
        stats = self.milvus_client.get_collection_stats(self.milvus_collection_name)
        return stats

    def get_vector_from_milvus(self, idx: List[int]) -> None:
        print(f"Getting vectors from Milvus with ids: {idx}")
        np_output_list, milvus_output_list = [], []

        milvus_output_list = self.milvus_client.get(
            collection_name=self.milvus_collection_name,
            ids=idx
        )

        print(len(milvus_output_list))

        #for vec in milvus_output_list:
            #np_output_list.append(vec["vector"])

        #self.milvus_processed_data = np_output_list
        self.milvus_processed_data = milvus_output_list

    def get_all_vectors_from_milvus(self) -> None:
        print(f"Getting all vectors from Milvus...")
        np_output_list = []

        # Получаем статистику коллекции
        stats = self.milvus_client.get_collection_stats(self.milvus_collection_name)
        row_count = stats['row_count']

        # Параметры для пагинации
        BATCH_SIZE = 1000  # Размер пакета для запроса
        total_fetched = 0

        # Запрашиваем векторы с использованием пагинации
        while total_fetched < row_count:
            milvus_output_list = self.milvus_client.query(
                collection_name=self.milvus_collection_name,
                limit=BATCH_SIZE,
                offset=total_fetched  # Увеличиваем смещение
            )

            # Извлекаем векторы
            for vec in milvus_output_list:
                np_output_list.append(vec["vector"])

            # Обновляем количество извлеченных векторов
            total_fetched += len(milvus_output_list)

        # Теперь у вас есть все векторы в np_output_list
        self.milvus_processed_data = np_output_list
        print(f"Fetched {total_fetched} vectors.")

    def vector_search_by_embedding(self, emb_vec: List[List[float]]) -> List[List[dict]]:
        res = self.milvus_client.search(
            collection_name=self.milvus_collection_name,
            anns_field="vector",
            data=emb_vec,
            limit=10,
            search_params={"metric_type": "COSINE", "params": {}}  # Search parameters
        )
        return res

    def vector_search_by_id(self, _inner_id: int) -> List[dict]:
        res = self.milvus_client.query(
            collection_name=self.milvus_collection_name,
            filter="inner_id = {}".format(_inner_id),
            output_fields=["*"]
        )

        return res

    def clusterize_data(self) -> None:
        print("Clustering data...")
        self.milvus_processed_data = np.array(self.milvus_processed_data)
        # Настройка K-means
        n_clusters = 3  # Укажите количество кластеров
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Обучение K-means
        kmeans.fit(self.milvus_processed_data)

        # Получите метки кластеров и центры
        self.clustered_labels = kmeans.labels_
        self.clustered_centers = kmeans.cluster_centers_


def main():
    test_obj = EmbeddingPipeline('test_md.xlsx')
    test_obj.import_df()
    test_obj.prepare_df()
    test_obj.process_df()
    test_obj.connect_to_milvus()
    test_obj.set_milvus_collection_name('test_collection')
    print(test_obj.get_milvus_collection_name())
    test_obj.drop_milvus_db()
    test_obj.create_milvus_db()
    test_obj.send_data_to_milvus()
    test_obj.get_vector_from_milvus([0])
    print(test_obj.get_milvus_processed_data())
    test_obj.get_vector_from_milvus([1])
    print(test_obj.get_milvus_processed_data())
    test_obj.get_vector_from_milvus([0, 1, 2, 3])
    print(test_obj.get_milvus_processed_data())
    print(test_obj.milvus_data_count())
    #print(test_obj.vector_search_by_id(253945))


main()
