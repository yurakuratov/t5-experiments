import tensorflow as tf
import os
import gzip
import json
import t5
import spacy

from tqdm import tqdm
import tensorflow_datasets as tfds

DATA_DIR = '/home/asagirova/hotpotqa'
nq_counts_path = os.path.join(DATA_DIR, "hotpotqa-counts.json")
nq_tsv_path = {
    "train": os.path.join(DATA_DIR, "hotpotqa-train-2250.tsv"),
    "validation": os.path.join(DATA_DIR, "hotpotqa-val-2250.tsv")
}

def nq_jsonl_to_tsv(in_fname, out_fname):

  nlp = spacy.blank("en")

  count = 0
  with open(in_fname, "rb") as infile,\
       open(out_fname, "w") as outfile:
    data = json.load(infile)
    for ex in tqdm(data):
      
      # Questions in NQ do not include a question mark.
      _id = ex['_id']
      question = ex["question"]
      answer = ex['answer']
      context = ' '.join(['. '.join([i[0], ''.join(i[1])]) for i in ex['context']])
      
      def word_tokenize(sent):
        doc = nlp(sent)
        return [token.text for token in doc]
      
      if len(word_tokenize(context)) > 2260:
        continue

      outfile.write("%s\t%s\t%s\t%s\n" % (_id, question, answer, context))
      count += 1
      
    return count

'''
# Create TSVs and get counts.
num_nq_examples = {}
num_nq_examples['train'] = nq_jsonl_to_tsv(
      os.path.join(DATA_DIR, 'hotpot_train_v1.1.json'), nq_tsv_path['train'])
num_nq_examples['validation'] = nq_jsonl_to_tsv(
      os.path.join(DATA_DIR, 'hotpot_dev_distractor_v1.json'), nq_tsv_path['validation'])

json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))
'''


def nq_dataset_fn(split, shuffle_files=False):
  
  import functools
  import os
  import time
  import warnings
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path[split])
  # Split each "<question>\t<answer>" example into (question, answer) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", "", "", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"question": ... "answer": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["id", "question", "answer", "context"], ex)))
  ds = ds.map(lambda ex: {"id": ex['id'], "question": ex['question'], "answers": { 'text': [ex['answer']] }, "context": ex['context']})
  return ds

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