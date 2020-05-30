import pickle
import os
import json
import time
import collections
import numpy as np
import eval_utils as util
import nq_eval as nq_eval
import tensorflow as tf

# flags = tf.flags
# FLAGS = flags.FLAGS

# Required parameters
# flags.DEFINE_string("predictions_dir", None,
                    # 'Path to directory containing predictions JSON.')

# flags.DEFINE_string("output_prediction_file", None,
                   # 'Path to output predictions JSON.')

# A input structure for storing prediction and annotation.
# When a example has multiple annotations, multiple NQLabel will be used.
NQLabel = util.collections.namedtuple(
    'NQLabel',
    [
        'example_id',  # the unique id for each NQ example.
        'long_answer_span',  # A Span object for long answer.
        'short_answer_span_list',  # A list of Spans for short answer.
        #   Note that In NQ, the short answers
        #   do not need to be in a single span.
        'yes_no_answer',  # Indicate if the short answer is an yes/no answer
        #   The possible values are "yes", "no", "none".
        #   (case insensitive)
        #   If the field is "yes", short_answer_span_list
        #   should be empty or only contain null spans.
        'long_score',  # The prediction score for the long answer prediction.
        'short_score'  # The prediction score for the short answer prediction.
    ])


class util_Span(object):
    """A class for handling token and byte spans.

      The logic is:

      1) if both start_byte !=  -1 and end_byte != -1 then the span is defined
         by byte offsets
      2) else, if start_token != -1 and end_token != -1 then the span is define
         by token offsets
      3) else, this is a null span.

      Null spans means that there is no (long or short) answers.
      If your systems only care about token spans rather than byte spans, set all
      byte spans to -1.

    """

    def __init__(self, start_byte, end_byte, start_token_idx, end_token_idx):

        if ((start_byte < 0 and end_byte >= 0) or
                (start_byte >= 0 and end_byte < 0)):
            raise ValueError('Inconsistent Null Spans (Byte).')

        if ((start_token_idx < 0 and end_token_idx >= 0) or
                (start_token_idx >= 0 and end_token_idx < 0)):
            raise ValueError('Inconsistent Null Spans (Token).')

        if start_byte >= 0 and end_byte >= 0 and start_byte >= end_byte:
            raise ValueError('Invalid byte spans (start_byte >= end_byte).')

        if ((start_token_idx >= 0 and end_token_idx >= 0) and
                (start_token_idx >= end_token_idx)):
            raise ValueError('Invalid token spans (start_token_idx >= end_token_idx)')

        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx


    def is_null_span(self):
        """A span is a null span if the start and end are both -1."""

        if (self.start_byte < 0 and self.end_byte < 0 and
                self.start_token_idx < 0 and self.end_token_idx < 0):
            return True
        return False


    def __str__(self):
        byte_str = 'byte: [' + str(self.start_byte) + ',' + str(self.end_byte) + ')'
        tok_str = ('tok: [' + str(self.start_token_idx) + ',' + str(
            self.end_token_idx) + ')')

        return byte_str + ' ' + tok_str

    def __repr__(self):
        return self.__str__()


class ScoreSummary(object):
  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


def sigmoid(x):
    """Applies logistic sigmoid function to x to return value within (0,1)"""
    return 1 / (1 + np.exp(-x))


def ensemble_nbest_to_nq_pred_dict(nbest_list, wts, yes_thr=None, no_thr=None):
    """Computes official answer key from raw logits across all prediction sets in ensemble"""
    # assert len(nbest_lst) == len(wts)
    nq_pred_dict = {}

    # for i in range(1, len(nbest_lst)):
    #     assert set(nbest_lst[0].keys()) == set(nbest_lst[i].keys())

    for example_id in nbest_list[0].keys():  # iterate through all examples

        d_long = {}  # key: (start_tok,end_tok), value: prob
        d_short = {}
        yes_logits, no_logits = 0, 0
        yes_exception, no_exception = False, False
        for i, nbest in enumerate(nbest_list):  # iterate through all models predictions and their values
            if yes_thr and (not yes_exception):
                try:
                    yes_logits += nbest[example_id][0].answer_type_logits[1]  # yes answer

                except:
                    print("exception yes_logits += nbest[example_id][0].answer_type_logits[1] * wts[i]")
                    yes_exception = True
            if no_thr and (not no_exception):
                try:
                    no_logits += nbest[example_id][0].answer_type_logits[2] * wts[i]  # no answer
                except:
                    print("exception no_logits += nbest[example_id][0].answer_type_logits[2] * wts[i]")
                    no_exception = True
            long_seen = set()
            short_seen = set()
            for score_summary in nbest[example_id]:  # iterate through scores in for example in current model
                pred_lbl = score_summary.predicted_label
                long_ans = pred_lbl['long_answer']
                short_ans = pred_lbl['short_answers'][0]
                long_key = long_ans['start_token'], long_ans['end_token']
                short_key = short_ans['start_token'], short_ans['end_token']
                if long_key not in long_seen:
                    long_seen.add(long_key)
                    if long_key not in d_long:
                        d_long[long_key] = sigmoid(pred_lbl['long_answer_score']) * wts[i]  # set sigmoid output
                    else:
                        d_long[long_key] += sigmoid(pred_lbl['long_answer_score']) * wts[i]  # accumulate sigmoid output
                if short_key not in short_seen:
                    short_seen.add(short_key)
                    if short_key not in d_short:
                        d_short[short_key] = sigmoid(pred_lbl['short_answers_score']) * wts[i]
                    else:
                        d_short[short_key] += sigmoid(pred_lbl['short_answers_score']) * wts[i]

        (start_tok_long, end_tok_long), prob_long = sorted(list(d_long.items()),
                                                           key=lambda x: x[1], reverse=True)[0]
        (start_tok_short, end_tok_short), prob_short = sorted(list(d_short.items()),
                                                              key=lambda x: x[1], reverse=True)[0]

        # Set final short answer span
        short_answer_span_list = [util_Span(-1, -1, start_tok_short, end_tok_short)]
        yes_no_answer = 'none'
        try:
            if yes_thr and yes_logits > yes_thr:
                yes_no_answer, prob_short = 'yes', 999999
                short_answer_span_list = [util_Span(-1, -1, -1, -1)]
            elif no_thr and no_logits > no_thr:
                yes_no_answer, prob_short = 'no', 999999
                short_answer_span_list = [util_Span(-1, -1, -1, -1)]
        except:
            print('exception if yes_thr and yes_logits > yes_thr ...')

        # Set final long answer span
        long_answer_span_list = util_Span(-1, -1, start_tok_long, end_tok_long)

        # Assemble prediction dictionary
        nq_pred_dict[int(example_id)] = NQLabel(
            example_id=example_id,
            long_answer_span=long_answer_span_list,
            short_answer_span_list=short_answer_span_list,
            yes_no_answer=yes_no_answer,
            long_score=prob_long,
            short_score=prob_short)
    return nq_pred_dict


def print_r_at_p_table(answer_stats, targets=[], thr_in=None):
    """Pretty prints the R@P table for default targets."""
    opt_result, pr_table = nq_eval.compute_pr_curves(
        answer_stats, targets=targets)
    f1, precision, recall, threshold = opt_result

    if thr_in: threshold = thr_in

    tp = sum([x[2] and x[3] >= threshold for x in answer_stats])
    true = sum([x[0] for x in answer_stats])
    pred = sum([x[1] and x[3] >= threshold for x in answer_stats])

    if not thr_in:
        print('Optimal threshold: {:.5}'.format(threshold))
        print(' F1     /  P      /  R')
        print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))
        for target, recall, precision, row in pr_table:
            print('R@P={}: {:.2%} (actual p={:.2%}, score threshold={:.4})'.format(
                target, recall, precision, row))
    else:
        precision = nq_eval.safe_divide(tp, pred)
        recall = nq_eval.safe_divide(tp, true)
        f1 = nq_eval.safe_divide(2 * precision * recall, precision + recall)
        print('Input threshold: {:.5}'.format(threshold))
        print(' F1     /  P      /  R')
        print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))

    return threshold, tp, true, pred, f1


def score_answers(gold_annotation_dict, pred_dict, thr_long=None, thr_short=None, sort_by_id=False):
    """Scores all answers for all documents.
    Args:
      gold_annotation_dict: a dict from example id to list of NQLabels.
      pred_dict: a dict from example id to list of NQLabels.
      sort_by_id: if True, don't compute F1; if False, compute F1 and print
    Returns:
      long_answer_stats: List of scores for long answers.
      short_answer_stats: List of scores for short answers.
    """
    # gold_annotation_dict = nq_gold_dict
    # pred_dict = nq_pred_dict

    gold_id_set = set(gold_annotation_dict.keys())
    pred_id_set = set(pred_dict.keys())

    if gold_id_set.symmetric_difference(pred_id_set):
        raise ValueError('ERROR: the example ids in gold annotations and example '
                         'ids in the prediction are not equal.')

    long_answer_stats = []
    short_answer_stats = []

    for example_id in gold_id_set:
        gold = gold_annotation_dict[example_id]
        pred = pred_dict[example_id]

        if sort_by_id:
            long_answer_stats.append(list(nq_eval.score_long_answer(gold, pred)) + [example_id])
            short_answer_stats.append(list(nq_eval.score_short_answer(gold, pred)) + [example_id])
        else:
            long_answer_stats.append(nq_eval.score_long_answer(gold, pred))
            short_answer_stats.append(nq_eval.score_short_answer(gold, pred))

    # use the 'score' column, which is last
    long_answer_stats.sort(key=lambda x: x[-1], reverse=True)
    short_answer_stats.sort(key=lambda x: x[-1], reverse=True)

    if not sort_by_id:
        print('-' * 20)
        print('LONG ANSWER R@P TABLE:')
        thr_long, tp_long, true_long, pred_long, f1_long = print_r_at_p_table(long_answer_stats, thr_in=thr_long)
        print('-' * 20)
        print('SHORT ANSWER R@P TABLE:')
        thr_short, tp_short, true_short, pred_short, f1_short = print_r_at_p_table(short_answer_stats, thr_in=thr_short)

        precision = nq_eval.safe_divide(tp_long + tp_short, pred_long + pred_short)
        recall = nq_eval.safe_divide(tp_long + tp_short, true_long + true_short)
        f1 = nq_eval.safe_divide(2 * precision * recall, precision + recall)

        print('-' * 20)
        print(' F1     /  P      /  R')
        print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))

        return long_answer_stats, short_answer_stats, thr_long, thr_short, f1_long, f1_short, f1
    return long_answer_stats, short_answer_stats

def nqlabel_to_dict(nqlabel):
    # nqlabel.example_id
    # nqlabel.long_answer_span
    # nqlabel.short_answer_span_list
    # nqlabel.yes_no_answer
    # nqlabel.long_score
    # nqlabel.short_score
    return {'example_id': int(nqlabel.example_id),
             'long_answer': {'start_byte': int(nqlabel.long_answer_span.start_byte),
                             'end_byte': int(nqlabel.long_answer_span.end_byte),
                             'start_token': int(nqlabel.long_answer_span.start_token_idx),
                             'end_token': int(nqlabel.long_answer_span.end_token_idx)},
             'long_answer_score': float(nqlabel.long_score),
             'short_answers': [{'start_byte': int(x.start_byte),
                                'end_byte': int(x.end_byte),
                                'start_token': int(x.start_token_idx),
                                'end_token': int(x.end_token_idx)} for x in nqlabel.short_answer_span_list],
             'short_answers_score': float(nqlabel.short_score),
             'yes_no_answer': nqlabel.yes_no_answer.upper()
            }


def get_nq_pred_dict(nbest_ens_list):
    ensemble_wts = [1, 1.1, 0.8]
    yes_thr = 5.5
    no_thr = 5.5

    # Add all pickle prediction files to list
    # pkl_list = []
    # for file in os.listdir(predictions_dir):
        # if file.endswith(".pkl"):
            # pkl_list.append(file)

    # Create list of all predictions across all models in ensemble (length 3)
    # nbest_ens_list = [pickle.load(open(predictions_dir + nbest_file, 'rb')) for nbest_file in pkl_list]

    # Create prediction dictionary in official NQ format using best scores for each example
    nq_pred_dict_ens = ensemble_nbest_to_nq_pred_dict(nbest_ens_list, ensemble_wts,
                                                      yes_thr, no_thr)

    # Create final prediction dictionary dropping NQLabels objects for each prediction
    final_nq_pred_dict = [nqlabel_to_dict(v)
                          for (k, v) in nq_pred_dict_ens.items()]

    return final_nq_pred_dict



if __name__ == "__main__":
    
    # time_start = time.time()

    # Define ensemble weights & answer type thresholds
    ensemble_wts = [1, 1.1, 0.8]
    yes_thr = 5.5
    no_thr = 5.5

    # Add all pickle prediction files to list
    # pkl_list = []
    # for file in os.listdir(FLAGS.predictions_dir):
        # if file.endswith(".pkl"):
            # pkl_list.append(file)

    # Create list of all predictions across all models in ensemble (length 3)
    # nbest_ens_list = [pickle.load(open(FLAGS.predictions_dir + nbest_file, 'rb'))
                      # for nbest_file in pkl_list]

    # Create prediction dictionary in official NQ format using best scores for each example
    # nq_pred_dict_ens = ensemble_nbest_to_nq_pred_dict(nbest_ens_list, ensemble_wts,
                                                     # yes_thr, no_thr)
    
    # Create final prediction dictionary dropping NQLabels objects for each prediction
    # final_nq_pred_dict = [nqlabel_to_dict(v)
                          # for (k, v) in nq_pred_dict_ens.items()]

    # final_predictions_json = {"predictions": list(final_nq_pred_dict)}  # format dictionary for JSON conversion
    
    # print('Open output_prediction_file:')
    # print(FLAGS.output_prediction_file)
    # with tf.gfile.Open(FLAGS.output_prediction_file, "w") as f:
        # json.dump(final_predictions_json, f, indent=4)
    
    # print('Post-processing complete: {} seconds'.format(time.time()-time_start))