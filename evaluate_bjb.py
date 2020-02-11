import pandas as pd
import json
# Read in predictions to data frame
test_answers_df = pd.read_json("predictions.json")
# Compute and select short / long answers based on confidence scores
def create_short_answer(entry):
    # if entry["short_answers_score"] < 1.5:
    #     return ""
    answer = []
    for short_answer in entry["short_answers"]:
        if short_answer["start_token"] > -1:
            answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
    if entry["yes_no_answer"] != "NONE":
        answer.append(entry["yes_no_answer"])
    return " ".join(answer)
def create_long_answer(entry):
    # if entry["long_answer_score"] < 1.5:
    # return ""
    answer = []
    if entry["long_answer"]["start_token"] > -1:
        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
    return " ".join(answer)
test_answers_df["long_answer_score"] = test_answers_df["predictions"].apply(lambda q: q["long_answer_score"])
test_answers_df["short_answer_score"] = test_answers_df["predictions"].apply(lambda q: q["short_answers_score"])
print(test_answers_df["long_answer_score"].describe())