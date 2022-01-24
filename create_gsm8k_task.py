import tensorflow as tf
import os
import gzip
import json
import t5

from tqdm import tqdm
from t5 import seqio
from t5.evaluation import qa_utils
from t5.data.tasks import DEFAULT_OUTPUT_FEATURES
import tensorflow_datasets as tfds


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("/home/asagirova/asc-trr/grade-school-math/grade_school_math/data/", f"{split}.jsonl")
    examples = read_jsonl(path)
    for i, ex in enumerate(examples):
        ex.update(_id=str(i))
        ex.update(question=ex["question"] + "\n")
        
        solutn, res = ex["answer"].split("####")
        ex.update(solution=solutn) #"<|startoftext|>" + 
        ex.update(result="####"+res#+"<|endoftext|>"
                 )
    print(f"{len(examples)} {split} examples")
    return examples



'''
DATA_DIR = '/home/asagirova/gsm8k-t5'
nq_tsv_path = {
    "train": os.path.join(DATA_DIR, "gsm8k-train.tsv"),
    "test": os.path.join(DATA_DIR, "gsm8k-test.tsv")
}

def nq_jsonl_to_tsv(split, out_fname):
  count = 0
  path = os.path.join("/home/asagirova/asc-trr/grade-school-math/grade_school_math/data/", f"{split}.jsonl")
  data = read_jsonl(path)
   
  with open(out_fname, "w") as outfile:
    for i, ex in tqdm(enumerate(data)):
      
      # Questions in NQ do not include a question mark.
      _id = i
      question = ex["question"] + "\n"
      solution, result = ex["answer"].split("####")
      solution = "<|startoftext|>" + solution
      result = "####" + result + "<|endoftext|>"
      
      outfile.write("%s\t%s\t%s\t%s\n" % (_id, question, solution, result))
      count += 1
      
    return count


# Create TSVs and get counts.
num_nq_examples = {}
num_nq_examples['train'] = nq_jsonl_to_tsv(
      os.path.join(DATA_DIR, 'hotpot_train_v1.1.json'), nq_tsv_path['train'])
num_nq_examples['validation'] = nq_jsonl_to_tsv(
      os.path.join(DATA_DIR, 'hotpot_dev_distractor_v1.json'), nq_tsv_path['validation'])

json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))
'''


def gsm_dataset_fn(split, shuffle_files=False):
  
  import functools
  import os
  import time
  import warnings
  # We only have one file for each split.
  del shuffle_files

  
  examples = get_examples(split)
  ids_str = [ex["_id"] for ex in tqdm(examples)]
  qns_str = [ex["question"] for ex in tqdm(examples)]
  sol_str = [ex["solution"] for ex in tqdm(examples)]
  res_str = [ex["result"] for ex in tqdm(examples)]
  print(f"\n\n\n len qns = {len(qns_str)}, len sol = {len(sol_str)}")
  ds = tf.data.Dataset.from_generator(lambda: zip(ids_str, qns_str, sol_str, res_str),
                                      output_types=(tf.string, tf.string, tf.string, tf.string),
                                      output_shapes=((), (), (), ())
                                     )
  '''
  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path[split])
  # Split each "<question>\t<answer>" example into (question, answer) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", "", "", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  '''
  # Map each tuple to a {"question": ... "answer": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["id", "question", "solution", "result"], ex)))
  ds = ds.map(lambda ex: {"id": ex['id'], "question": ex['question'], "solution": ex['solution'], "result": ex['result']})
  return ds


@seqio.map_over_dataset
def gsm8k(x, include_context=False):
  from t5.data.preprocessors import _string_join
  """Convert GSM examples to a text2text pair.
  GSM produces examples with this form:
    {'id': <id>, 'question': <problem>, solution': <detailed solution>,
     'result': <resulting answer>}
  This function will return examples of the format:
    {'inputs': 'question: <question> solution: <article>',
     'targets': '<result>',
     'id': <id>, 'question': <question>, 'solution': <solution>,
     'result': <result>},
  Args:
    x: an example to process.
    include_context: a boolean
  Returns:
    A preprocessed example with the format listed above.
  """
  def _gsm_pad_punctuation(text):
    """Adds spaces around punctuation."""
    # Add space around punctuation.
    text = tf.strings.regex_replace(text, r'([^0-9A-Za-z#_])', r' \1 ')
    # Collapse consecutive whitespace into one space.
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    return text
  
  print(f"\n\n\nx keys {x.keys()}")
  q = _gsm_pad_punctuation(x['question'])
  s = _gsm_pad_punctuation(x['solution'])
  r = _gsm_pad_punctuation(x['result'])
  if include_context:
    inputs = _string_join(['question:', q, 'solution:', s])
    targets = r
  else:
    inputs = _string_join(['question:', q])
    targets = s + r
  return {
      'inputs': inputs,
      'targets': targets,
      'id': x['id'],
      'question': q,
      'solution': s,
      'result': r
  }

#print("A few raw validation examples...")
#for ex in tfds.as_numpy(nq_dataset_fn("train").take(5)):
#  print(ex)

'''
def trivia_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(#'question: <question> context: <article>'
                 ["question: ", normalize_text(ex["question"]),"context: ", normalize_text(ex["context"])]),
        "targets": normalize_text(ex["answer"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                


from create_hotpotqa_task import nq_dataset_fn, num_nq_examples, trivia_preprocessor
t5.data.TaskRegistry.add(
    "hotpotqa_context",
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=nq_dataset_fn,
    splits=["train"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[trivia_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy]#,
    # Not required, but helps for mixing and auto-caching.
    #num_input_examples=num_nq_examples
)

nq_task = t5.data.TaskRegistry.get("hotpotqa_context")
ds = nq_task.get_dataset(split="train", sequence_length={"inputs": 128, "targets": 32})
'''
#print("A few preprocessed validation examples...")
#for ex in tfds.as_numpy(ds.take(5)):
#  print(ex)
def test_solve_rate(targets, predictions):
  """Computes SQuAD metrics, maximizing over answers per question.
  Args:
    targets: list of lists of results
    predictions: list of predicted solutions
  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [qa_utils.normalize_squad(t)for t in targets]
  predictions = [qa_utils.normalize_squad(p) for p in predictions]
  
  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")
  overall = len(targets)
  c = 0
  for t, p in zip(targets, predictions):
    
    # t5.data.get_default_vocabulary().decode([3, 30345,30345]) = "####"
    final_answer_start = tf.where(t == 30345)[-1][0]
    final_answer_end = tf.where(t == 1)[0][0]
    real_res = t[final_answer_start+1:final_answer_end]
  
    
    zz = tf.where(p == 30345)
    if zz:
      pred_answer_start = zz[-1][0]
      ozzy = tf.where(p == 1)
      if ozzy:
        pred_answer_end = ozzy[0][0]
        pred_res = p[pred_answer_start+1:pred_answer_end]
        c += int(real_res == pred_res)
      else:
        logging.info(f'/n/npred answer has no eos token: {p}/n/n')
    else:
        logging.info(f'/n/npred answer has no #### token: {p}/n/n')
        
  
  tsr = c / overall
  tsr *= 100
  logging.info("Test Solve Rate = %.2f", tsr)
  return {"tsr": tsr}

"""
import functools
t5.data.TaskRegistry.add(
    "gsm8k_baseline",
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=gsm_dataset_fn,
    splits=["train", "test"],
    text_preprocessor=[
        functools.partial(gsm8k, include_context=True),
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[t5.evaluation.metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)
    
t5.data.TaskRegistry.add(
    "gsm8k_wm",
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=gsm_dataset_fn,
    splits=["train", "test"],
    text_preprocessor=[
        functools.partial(gsm8k, include_context=False),
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[t5.evaluation.metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)
"""