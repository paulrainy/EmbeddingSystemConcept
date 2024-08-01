import pandas as pd
import numpy as np
import torch
import math
from torch import Tensor
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer, AutoModel


class EmbeddingPipeline:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.df = self.import_df(self.file_path)
        self.processed_data = self.prepare_df()

    def get_file_path(self) -> str:
        return self.file_path

    @staticmethod
    def import_df(path: str) -> pd.DataFrame:
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise ValueError(f"Error importing file: {e}")

    def prepare_df(self) -> List[pd.DataFrame]:
        df = self.import_df(self.file_path)

        def remove_columns_step(dataset: pd.DataFrame) -> pd.DataFrame:
            output_dataset = dataset[
                ["Id", "Direction", "Section", "TestCaseName", "Preconditions", "Steps", "Postconditions",
                 "ExpectedResult"]].copy()
            output_dataset[["Id", "Direction", "Section", "TestCaseName"]] = output_dataset[
                ["Id", "Direction", "Section", "TestCaseName"]].ffill()
            return output_dataset

        output_first_step = remove_columns_step(df)

        def parse_tests_by_id(dataset: pd.DataFrame) -> List[pd.DataFrame]:
            test_cases = []
            for group_name, frame in dataset.groupby('Id'):
                test_cases.append(frame)

            return test_cases

        output_second_step = parse_tests_by_id(output_first_step)

        def remove_empty_cells(array: List[pd.DataFrame]) -> List[pd.DataFrame]:
            def up_cells(tst_case: pd.DataFrame, column: str) -> pd.DataFrame:
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

        return output_third_step

    def process_df(self, df) -> List[Tensor]:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        def mean_pooling(model_output, attention_mask) -> Tensor:
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        mpnet_vectorized_output_array = []

        model.to(device)

        for test_case in df:
            # Tokenize
            encoded_input = tokenizer(str(test_case), return_tensors='pt', padding=True, truncation=True,
                                      max_length=512).to(device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling. In this case, average pooling
            embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            mpnet_vectorized_output_array.append(F.normalize(embedding, p=2, dim=1))

        return mpnet_vectorized_output_array
