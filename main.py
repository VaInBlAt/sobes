import pandas as pd
import nli, paraphrase

df = pd.read_csv("test/test.csv", sep=";")

answers1 = df["correct_answer"]
answers2 = df["answer1"]

df["compare_nli"] = df.apply(nli.compare_as_row_nli, axis=1)
df["compare_paraphrase"] = df.apply(paraphrase.compare_as_row_paraphrase, axis=1)

NLI_THRESHOLD = 0.8
PARAPHRASE_THRESHOLD = 0.8

df["good_nli"] = df.apply(
    lambda row: nli.is_good_answer_nli(row, NLI_THRESHOLD), 
    axis=1
)
df["good_paraphrase"] = df.apply(
    lambda row: paraphrase.is_good_answer_paraphrase(row, PARAPHRASE_THRESHOLD), 
    axis=1
)

print(df[["good_nli", "good_paraphrase"]])
