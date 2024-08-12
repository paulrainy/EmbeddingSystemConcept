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

        def remove_columns(dataset: List[DataFrame]) -> List[DataFrame]:
            array = []
            for dframe in dataset:
                array.append(dframe[["Direction", "TestCaseName", "Steps", "ExpectedResult"]])

            return array

        prepared_df_without_id = remove_columns(self.prepared_df)

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

    def connect_to_milvus_client(self) -> None:
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

    def create_milvus_collection(self) -> None:
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
            schema.add_field(field_name="direction_name", datatype=DataType.VARCHAR, max_length=1024)
            # section of test case
            schema.add_field(field_name="section_name", datatype=DataType.VARCHAR, max_length=1024)
            # name of test case
            schema.add_field(field_name="test_case_name", datatype=DataType.VARCHAR, max_length=1024)
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

    def drop_milvus_collection(self) -> None:
        try:
            self.milvus_client.drop_collection(
                collection_name=self.milvus_collection_name
            )
            print(f"Collection '{self.milvus_collection_name}' dropped successfully!")
        except Exception as e:
            print(f"An error occurred while dropping the collection '{self.milvus_collection_name}': {e}")

    def send_data_to_milvus_collection(self) -> None:
        print("Sending data to Milvus")
        np_input_list = []
        for t_outer in self.local_processed_data:
            for t_inner in t_outer:
                np_input_list.append(t_inner.cpu().numpy())

        print(f"Vectors count: {len(np_input_list)}")

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

    def get_vectors_by_idx(self, idx: List[int]) -> None:
        print(f"Getting vectors from Milvus with ids: {idx}")
        milvus_output_list = self.milvus_client.get(
            collection_name=self.milvus_collection_name,
            ids=idx
        )
        print(f"Vector count: {len(milvus_output_list)}")
        self.milvus_processed_data = milvus_output_list

    def search_vectors_by_embedding(self, emb_vec: List[List[float]]) -> List[List[dict]]:
        res = self.milvus_client.search(
            collection_name=self.milvus_collection_name,
            anns_field="vector",
            data=emb_vec,
            limit=10,
            output_fields=["vector", "inner_id"],
            search_params={"metric_type": "COSINE", "params": {}}  # Search parameters
        )
        return res

    def search_vector_by_inner_id(self, _inner_id: int) -> List[dict]:
        res = self.milvus_client.query(
            collection_name=self.milvus_collection_name,
            filter="inner_id == {}".format(_inner_id),
            output_fields=["vector", "inner_id"]
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
    test_obj = EmbeddingPipeline('test_lg.xlsx')
    test_obj.import_df()
    test_obj.prepare_df()
    test_obj.process_df()
    test_obj.connect_to_milvus_client()
    test_obj.set_milvus_collection_name('test_collection')
    test_obj.drop_milvus_collection()
    test_obj.create_milvus_collection()
    test_obj.send_data_to_milvus_collection()

    print(test_obj.search_vector_by_inner_id(253945))
    print(test_obj.search_vectors_by_embedding([[0.004513017, -0.042707697, -0.006404353, 0.015330272, 0.028008759, -0.003062166, -0.014132557, 0.008616002, -0.0040410087, 0.07452528, 0.08257223, 0.0037822428, 0.008011247, 0.082423575, -0.021442765, -0.059667833, -0.0011082964, 0.0050324397, -0.07160051, 0.03327071, 0.0352759, 0.026264915, 0.0019115376, -0.001120512, -0.012997476, 0.010937711, -0.0051716184, 0.03333103, 0.03955518, 0.03182424, 0.06898222, 0.015261671, 0.012830509, -0.059578497, 0.007132644, 0.01038468, -0.03608632, -0.0022819692, 0.020465955, 0.017011294, 0.07127496, -0.047913264, -0.025456829, -0.0059506027, -0.03904922, -0.031856198, -0.017426139, 0.07038733, 0.014350913, 0.026212648, -0.0021413257, 0.05014284, -0.015194753, 0.0388937, 0.0465326, -0.10837583, -0.015068561, 0.064096004, -0.059954192, -0.0005247501, 0.015792068, 0.012588785, 0.041390505, 0.010434706, -0.017975567, -0.013976275, 0.06475284, 0.0029282197, -0.05824965, -0.01992101, 0.10582185, 0.061919305, 0.0014808819, 0.015223062, -0.015933676, -0.033087138, -0.021606062, -0.04870316, -0.009084844, -0.0042229574, 0.08083623, 0.05068407, -0.0580899, -0.05810537, -0.013250752, 0.023325961, 0.006298164, 0.0023194482, -0.0255736, 0.035913203, -0.02229454, -0.036341384, 0.019370476, -0.00036267008, -0.022509335, 0.04247368, -0.0061953454, 0.013010369, 0.019187089, -0.010419719, 0.01555414, 0.04299227, 0.026968732, 0.031262714, -0.0021784413, -0.027834088, -0.062960826, 0.0075454875, -0.021734508, 0.0333921, -0.0009810206, -0.019133503, -0.036929462, -0.004608932, 0.023910461, -0.013168195, -0.0069454475, 0.02606872, -0.016638448, -0.015733924, 0.11758507, -0.038894676, -0.044557977, -0.002438712, -0.022314185, 0.0056465575, 0.01531968, 0.0017908475, 0.0087159285, 0.05116786, 0.003340493, -0.054578938, -0.033268385, -0.009657375, -0.031079423, 0.028769339, 0.0009998155, -0.058604404, -0.03166205, -0.059245728, 0.0045084343, -0.016851686, -0.014064526, -0.0016286487, -0.056086782, -0.0072678938, -0.02545672, 0.07714309, 0.0052824444, 0.010641322, 0.03595741, 0.0153076975, 0.018810261, 0.0068180254, 0.0025990813, 0.056378607, -0.02426361, -0.010318171, -0.07869885, -0.033017993, -0.057143483, -0.016793724, -0.009593102, 0.027089069, 0.037586108, 0.03386465, -0.036765616, 0.0035564872, 0.011822651, -0.039039105, -0.04702563, 0.039846268, 0.02568503, 0.013158406, -0.016351955, 0.07485551, 0.04321391, -0.020410242, 0.010280403, -0.018608548, -0.04055375, 0.008780742, 0.026878033, 0.023788108, -0.03827444, 0.010496911, -0.069538586, 0.003067567, -0.02938763, -0.0021245375, 0.0029233468, -0.020465052, -0.02609113, 0.0060532647, 0.05066956, 0.0009930296, -0.053285286, -0.12635484, 0.023226837, 0.026341239, -0.0024549363, -0.04557201, 0.03809012, 0.02093746, 0.0069520054, -0.042948898, -0.009227047, 0.0036890288, -0.0066315043, 0.025370141, 0.1290686, 0.03806978, 0.00420139, -0.022233145, -0.037901167, -0.015442884, 0.013005767, -0.0064345505, -0.087619886, -0.012938321, -0.03207493, 0.01661345, -0.06278763, 0.02004999, 0.01541879, 0.033734582, -0.024353534, 0.019454291, -0.026739806, 0.011684542, 0.015914408, 0.021881651, 0.017699236, -0.008152466, -0.015329545, 0.006176931, -0.026426109, 0.06478306, 0.011502297, -0.014867791, -0.06409388, 0.009607232, -0.023083957, -0.009035918, 0.069599934, -0.005460663, 0.028310314, 0.011898724, 0.02458535, 0.011199966, 0.008019882, 0.023582956, 0.012783096, -0.014586658, 0.037203338, 0.040145993, -0.07458049, 0.021191558, -0.015638804, -0.0040771225, 0.02462726, 0.0023768072, -0.037575394, -0.040472142, 0.018454967, -0.003558438, 0.029131133, -0.0036726599, -0.002612224, 0.019279785, 0.01716597, 0.04567345, -0.048082903, -0.043859478, 0.00890722, -0.0075259614, 0.005765499, 0.022608798, -0.041770853, 0.033063125, 0.0059964815, -0.035537638, -0.013418615, -0.034434967, -0.05631914, 0.036761314, 0.022091657, -0.029842813, -0.01636488, 0.03892148, 0.012952596, -0.0027647093, 0.012018717, -0.026911508, -0.034355313, -0.027608808, -0.0063322145, -0.03606221, -0.0038779874, 0.016890783, 0.0009949426, 0.0009778946, -0.016914794, 0.009903196, -0.030131616, 0.03314789, 0.023907656, -0.028136432, 0.07096195, 0.045339666, 0.025623864, 0.009433451, -0.0014336332, 0.027488993, -0.01434334, -0.004582762, 0.020775907, 0.006640788, 0.013016867, 0.06629692, 0.018720986, -0.0124293165, 0.012795235, 0.007158264, -0.0023134386, -0.06120191, -0.0530873, -0.0062832753, 0.042505767, -0.022548143, 0.02885648, 0.0024977406, 0.0160049, -0.013144409, -0.013539603, -0.05990787, 0.0008415051, 0.06618604, -0.012874405, 0.009347419, -0.029939065, -0.010665589, -0.0032497428, 0.021481778, 0.0497143, 0.029117644, 0.031879343, 0.018726766, 0.03515662, 0.010186555, 0.024644338, 0.01475904, -0.02561839, -0.011437896, 0.010283396, -0.041536592, 0.034255866, 0.015565293, -0.05337454, 0.013518791, 0.025961272, -0.018032586, -0.009263031, 0.01265037, -0.057818986, -0.03394054, -0.005100551, 0.046561047, -0.00052519987, 0.015704647, 0.050856385, 0.0023288853, 0.004376771, 0.0075305286, 0.021999456, 0.02008923, 0.003471431, -0.017346496, -0.03446604, 0.0021785696, 0.03618912, -0.05367441, 0.17454892, -0.016448256, 0.0040920633, -0.0044550714, -0.0039386004, -0.017895347, -0.019144204, 0.08323338, -0.02112303, -0.057639226, -0.087600745, -0.0030857273, 0.004711161, 0.039104734, -0.024021335, 0.057440866, 0.01836935, 0.0017463913, 0.029892193, -0.01773623, -0.052264832, -0.016458401, 0.034667093, 0.009501102, -0.010560929, -0.028557962, -0.016373528, -0.05255553, 0.04068802, -0.0412085, 0.016574234, -0.014894273, 0.012595751, 0.03349597, -0.030475315, -0.0443948, -0.0052970317, -0.03163241, 0.025546992, 0.015351664, 0.049570188, -0.032731768, -0.037563726, -0.03149396, 0.057863668, 0.05034888, 0.020741632, -0.033110455, -0.0005428865, -0.002122844, -0.0729027, 0.098436885, -0.06067086, -0.036652282, 0.027976092, 0.0028484408, 0.007742979, -0.03675433, -0.03751896, 0.009567767, 0.15169685, -0.028487803, -0.019413443, 0.010576218, 0.011656727, -0.03789792, -0.057148848, 0.069221355, 0.00763534, -0.017443113, -0.03657788, -0.017461551, 0.016600711, -0.011015396, 0.055873144, -0.04231921, -0.033032406, 0.0361672, 0.010656571, 0.056381892, -0.09028945, -0.028080815, 0.0152785275, -0.12177115, 0.013450071, 0.015315139, 0.0028921177, -0.032688253, 0.030615933, -0.010942722, -0.0048246295, 0.03568876, 0.02942505, 0.031438533, -0.009945213, -0.061994273, 0.06524909, 0.013284434, 0.039137695, 0.05276338, 0.014358855, -0.07610047, -0.01933869, -0.043974537, -0.009721887, -0.0062776683, -0.012131991, -0.0134125855, 0.022062398, -0.0035880595, -0.01295722, 0.03533252, -0.039063726, -0.004682649, 0.07147781, -0.023311963, 0.06657237, -0.012547908, 0.0048835897, -0.0034798982, 0.02009821, -0.03531657, -0.019884031, 0.07467717, 0.030256065, 0.01864412, 0.014519806, -0.038030066, 0.010779276, 0.02836386, -0.043549374, -0.10678932, -0.003012419, 0.02573428, -0.019012805, 0.0136833675, -0.004957116, -0.034023322, 0.027477587, 0.0015943532, -0.0025884493, -0.018079108, 0.06845845, 0.056036957, 0.02190874, -0.008260738, -0.01832676, -0.009682969, -0.067289904, -0.009408031, -0.0053051566, -0.010402141, 0.0014097182, 0.04506106, 0.06280382, -0.015081124, -0.039765213, -0.018619018, -0.02562938, 0.045807157, 0.012224979, 0.036057852, -0.015033701, -0.019354511, 0.01458753, -0.03294052, 0.02237309, 0.023310492, 0.019734336, -0.037661612, 0.0028894674, 0.012569028, 0.0132708335, -0.023673974, 0.011363778, -0.0047875815, 0.021416355, -0.028533945, 0.013578855, -0.01536506, -0.01873566, 0.03514552, -0.05223977, -0.042218547, 0.00826394, -0.019731494, 0.016010996, 0.0075337687, 0.021539118, -0.031315334, -0.02356848, 0.0007689896, -0.01186194, -0.019418865, 0.032696187, 0.0252106, 0.03382684, -0.04925183, 0.011762749, -0.0049116127, 0.017438393, 0.008166993, -0.06791352, -0.015359609, -0.011760197, -0.025479741, -0.018863244, 0.0335627, -0.042872824, 0.0046909596, 0.003292946, -0.009602178, -0.029734042, -0.0064987815, -0.01192373, 0.04005112, 0.017373642, -0.004599265, 0.03443058, -0.06556425, 0.0020671443, 0.014571069, 0.020045921, 0.033727277, 0.016455417, 0.03072594, -0.04277727, 0.010469897, -0.061108235, -0.02500206, 0.01269194, -0.07987923, 0.010087471, -0.007918703, 0.016913345, 0.031760436, 0.010160215, -0.0054903724, -0.0028536315, -0.012216658, 0.003539157, 0.10169164, -0.03201694, -0.011637478, 0.033049397, 0.004295162, -0.09159131, -0.03124119, 0.06830551, -0.046773206, -0.02464427, -0.0018579625, 0.020734934, 0.08102306, -0.022744508, 0.025370471, -0.027677698, -0.0009099621, 0.01912102, -0.039024465, -0.037486088, -0.04293199, 0.025200482, -0.027629074, -0.0013887159, 0.06703003, 0.017816706, 0.03587606, 0.06647985, 0.017527826, 0.017957043, 0.0016104903, -0.02885316, -0.021243142, 0.012414444, 0.022859123, 0.011971554, -0.02595609, -0.009823583, 0.0029959637, -0.032357037, -0.009552806, 0.01686373, -0.020620173, -0.021465955, 0.036250774, 0.023900557, 0.056012414, 0.0013238718, 0.021690276, -0.002933082, 0.010615917, 0.006666972, -0.013450366, -0.024837522, 0.06668517, 0.01893154, 0.009437161, -0.00299768, -0.009335431, 0.0015093442, 0.051279116, 0.027119208, -0.014404436, -0.000733439, 0.049793024, 0.020157175, 0.040509604, 0.0047338326, 0.054791432, 0.1024015, -0.033195008, -0.07422286, -0.034459636, 0.01581494, 0.019489922, -0.005324796, 0.053695712, -0.029341072, -0.034125272, 0.040695827, 0.007896052, -0.056492977, 0.03205783, -0.020915858, -0.03608907, 0.004382653, 0.015683249, 0.017666886, 0.030895477, -0.039232533, -0.015786001, 0.029769821, 0.0051122014, -0.041927107, 0.041498177, -0.0054967618, -0.009163936, 0.061931454, -0.015030946, 0.09199408, 0.028493477, -0.0021959553, -0.017828312, 0.070024826, 0.031715337, -0.030527562, -0.0127186915, 0.034480963, -0.005014779, 0.024530914, 0.0066885464, -0.08049913, 0.011722422, 0.117967464, -0.01815295, -0.023761703, 0.03857535, -0.04839244, 0.058773383, 0.054027677, -0.009807564, -0.023869378, -0.0077983392, 0.015810195, 0.0045014913, 0.09819446, -0.014018798, -0.015865719, 0.022844777, 0.023236083, -0.0034461138, -0.034148186, -0.016067643, 0.024911879, -0.00879732, 0.032012675, 0.011395462, 0.0015790263, 0.03572498, -0.025649732, -0.084779434, -0.0059847054, 0.054741662, -0.08282464, -0.06437031]]))
    # сделать более юзер френдли метод +
    # section name удалить +
    # поиск по функциональным областям (Игорь скинет)


main()
