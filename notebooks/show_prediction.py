# coding=utf-8
# @author John Bica | Hannah Stoik | Geoff Korb
"""
Utility function for displaying predictions' text span
"""

import jsonlines
import json
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

class ShowPrediction:
    """Class to store and access input needed for displaying prediction text spans"""
    def __init__(self, jsonl_file):
        self._data = {}
        with open(jsonl_file, 'rb') as f:
            for line in tqdm(f.readlines()):
                d = json.loads(line)
                self._data[int(d['example_id'])] = {
                    'doc_tokens': d['document_tokens'],
                    'text': " ".join([_clean_token(t) for t in d["document_tokens"]]),  # re-assemble text using tokens
                    'question': d['question_text']
                }

    def __call__(self, prediction, include_full_text=False, remove_html=True):
        """Returns the respective text for a prediction including question, long answer, short answer,
        and optional document text.
        There are also a very few unicode characters that are prepended with blanks.

        Args:
            prediction: Dictionary representation of prediction
            include_full_text: boolean to include full Wikipedia document text
            remove_html: boolean to include html tags

        Returns:
            Three (optional four) strings in dictionary representation of question, document text, long/short answer
        """
        data = self._data[prediction['example_id']]
        result = {'question': data['question']}
        if include_full_text:
            result['text'] = data['text']
        for type_ in ['long_answer', 'short_answers']:
            ans = prediction[type_]
            if isinstance(ans, list):
                ans = ans[0]
            start, end = ans['start_token'], ans['end_token']
            if remove_html:
                answer_span = [
                    item['token']
                    for item in data['doc_tokens'][start:end]
                    if not item['html_token']
                ]
            else:
                answer_span = [
                    item['token']
                    for item in data['doc_tokens'][start:end]
                ]
            result[type_] = " ".join(answer_span)
        return result


# HELPER FUNCTIONS #

def _clean_token(token):
    """Returns token in which blanks are replaced with underscores.
    HTML table cell openers may contain blanks if they span multiple columns.
    There are also a very few unicode characters that are prepended with blanks.

    Args:
        token: Dictionary representation of token in original NQ format.

    Returns:
        String token.
    """
    return re.sub(u" ", "_", token["token"])


def create_short_answer(entry):
    """Returns string token of the range of a short answer from start to end token
    (i.e : '23:26'), Yes/No, or empty string if no answer.

    Args:
        entry: Dictionary representation of prediction values for a particular example

    Returns:
        String of start and end tokens for short answer, 'Yes/No' or empty string if no answer
    """
    answer = []
    for short_answer in entry["short_answers"]:
        if short_answer["start_token"] > -1:
            answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
    if entry["yes_no_answer"] != "NONE":
        answer.append(entry["yes_no_answer"])
    return " ".join(answer)


def create_long_answer(entry):
    """Returns string token of the range of a long answer from start to end token
       (i.e : '20:58') or empty string if no answer.

    Args:
        entry: Dictionary representation of prediction values for a particular example

    Returns:
        String of start and end tokens for long answer or empty string if no answer
    """
    answer = []
    if entry["long_answer"]["start_token"] > -1:
        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
    return " ".join(answer)


def get_span_from_token_offsets(f, start_token_la, end_token_la, start_token_sa, end_token_sa, qas_id, remove_html):
    """Returns the respective text for a prediction including question, long answer, short answer,
    and optional document text.
    There are also a very few unicode characters that are prepended with blanks.

    Args:
        f: json input file
        start_token_la: integer value of token where long answer begins
        end_token_la: integer value of token where long answer ends
        start_token_sa: integer value of token where short answer begins
        end_token_sa: integer value of token where short answer ends
        qas_id: example id for the prediction
        remove_html: boolean to include html tags

    Returns:
        Question and text-representation of predicted answer.
    """
    for item in f:
        if item['example_id'] != qas_id:
            continue
        question = item['question_text']
        if remove_html:
            long_answer_span = [
                item['token']
                for item in item['document_tokens'][start_token_la:end_token_la]
                if not item['html_token']
            ]
            short_answer_span = [
                item['token']
                for item in item['document_tokens'][start_token_sa:end_token_sa]
                if not item['html_token']
            ]
        else:
            long_answer_span = [
                item['token']
                for item in item['document_tokens'][start_token_la:end_token_la]
            ]
            short_answer_span = [
                item['token']
                for item in item['document_tokens'][start_token_sa:end_token_sa]
                if not item['html_token']
            ]
        concat_long_answer_span = " ".join(long_answer_span)         # concatenate long answer tokens
        concat_short_answer_span = " ".join(short_answer_span)        # concatenate short answer tokens
        return question, concat_long_answer_span, concat_short_answer_span



