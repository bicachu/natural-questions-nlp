import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tf2_0_baseline_w_bert as tf2baseline
import bert_modeling as modeling
import bert_optimization as optimization
import bert_tokenization as tokenization
import json
import os
tf.compat.v1.disable_eager_execution()

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.compat.v1.app.flags.FLAGS)

flags = tf.compat.v1.app.flags

flags.DEFINE_string(
    "bert_config_file", "bertjointbaseline/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "bertjointbaseline/vocab-nq.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "outdir",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")

flags.DEFINE_string(
    "output_prediction_file", "predictions.json",
    "Where to print predictions in NQ prediction format, to be passed to"
    "natural_questions.nq_eval.")

flags.DEFINE_string(
    "init_checkpoint", "bertjointbaseline/bert_joint.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "verbosity", 1, "How verbose our error messages should be")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal NQ evaluation.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")


## Special flags - do not change

flags.DEFINE_string(
    "predict_file", "data/simplified-nq-test.jsonl",
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
flags.DEFINE_boolean("logtostderr", True, "Logs to stderr")
flags.DEFINE_boolean("undefok", True, "it's okay to be undefined")
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('HistoryManager.hist_file', '', 'kernel')

FLAGS = flags.FLAGS

# from here modeling

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

tf2baseline.validate_flags_or_throw(bert_config)
tf.io.gfile.makedirs(FLAGS.output_dir)

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

tpu_cluster_resolver = None
if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.compat.v1.estimator.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))

num_train_steps = None
num_warmup_steps = None

model_fn = tf2baseline.model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu)

# If TPU is not available, this falls back to normal Estimator on CPU or GPU.
estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    predict_batch_size=FLAGS.predict_batch_size)

if FLAGS.do_predict:
    if not FLAGS.output_prediction_file:
        raise ValueError(
            "--output_prediction_file must be defined in predict mode.")

    eval_examples = tf2baseline.read_nq_examples(
        input_file=FLAGS.predict_file, is_training=False)

    print("FLAGS.predict_file", FLAGS.predict_file)

    eval_writer = tf2baseline.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []


    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)


    num_spans_to_ids = tf2baseline.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()
    eval_filename = eval_writer.filename

    print("***** Running predictions *****")
    print(f"  Num orig examples = %d" % len(eval_examples))
    print(f"  Num split examples = %d" % len(eval_features))
    print(f"  Batch size = %d" % FLAGS.predict_batch_size)
    for spans, ids in num_spans_to_ids.items():
        print(f"  Num split into %d = %d" % (spans, len(ids)))

    predict_input_fn = tf2baseline.input_fn_builder(
        input_file=eval_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    print(eval_filename)

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []

    for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
        if len(all_results) % 1000 == 0:
            print("Processing example: %d" % (len(all_results)))

        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]

        all_results.append(
            tf2baseline.RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                answer_type_logits=answer_type_logits))

    print("Going to candidates file")

    candidates_dict = tf2baseline.read_candidates(FLAGS.predict_file)

    print("setting up eval features")

    eval_features = [
        tf.train.Example.FromString(r)
        for r in tf.compat.v1.python_io.tf_record_iterator(eval_filename)
    ]

    print("compute_pred_dict")

    nq_pred_dict = tf2baseline.compute_pred_dict(candidates_dict, eval_features,
                                                 [r._asdict() for r in all_results])
    predictions_json = {"predictions": list(nq_pred_dict.values())}

    print("writing json")

    with tf.io.gfile.GFile(FLAGS.output_prediction_file, "w") as f:
        json.dump(predictions_json, f, indent=4)