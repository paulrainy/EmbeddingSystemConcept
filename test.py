import pandas as pd
import math

df = pd.read_excel('test_lg.xlsx')

output_dataset = df[
    ["Id", "Direction", "Section", "TestCaseName", "Automated", "Preconditions", "Steps", "Postconditions",
     "ExpectedResult"]].copy()
output_dataset[["Id", "Direction", "Section", "TestCaseName", "Automated"]] = output_dataset[
    ["Id", "Direction", "Section", "TestCaseName", "Automated"]].ffill()

output_dataset.to_excel("first_output.xlsx", index=False)

test_cases = []
for group_name, frame in output_dataset.groupby('Id'):
    test_cases.append(frame)

array = pd.concat(test_cases, axis=0)
array.to_excel("second_output.xlsx", index=False)

# not working
for dframe in array:
    for col in ["Preconditions", "Steps", "Postconditions", "ExpectedResult"]:
        dframe[col].fillna(method='ffill', inplace=True)

    dframe["Steps"].fillna(dframe["Preconditions"], inplace=True)
    dframe["Steps"].fillna(dframe["Postconditions"], inplace=True)
    dframe.drop(["Preconditions", "Postconditions"], axis=1, inplace=True)

array.to_excel("third_output.xlsx", index=False)