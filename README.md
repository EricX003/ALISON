# ALISON
Fast and Effective Stylometric Authorship Obfuscation

## Data
The three datasets used can be found in the Data folder. Due to TuringBench's large size, it has been partitioned into three files.

## Prerequisites
Required packages are listed in the requirements.txt file:
```
pip install -r requirements.txt
```

## How to Use

* Train the n-gram based neural network classifier

```
python Train.py
```
Here we explain each argument in detail:

  * --train: The path to the training data.
  * --authors_total: Total number of authors in the dataset (default 10)
  * --dir: Path to the directory containing the trained model (Contains feature set, model, etc.)
  * --trial_name: Name of the trial (human-readable) to generate the save directory (default is empty string)
  * --test_size: Proportion of Data to use for network testing (default 0.15)
  * --top_ngrams: t, The Number of top character and POS-ngrams to retain
  * --V: V, the set of n-gram lengths to use (default '[1, 2, 3, 4]')

Additional arguments to fine-tune the training of the n-gram based neural network model are provided, and can be accessed via:
 ```
python Train.py -h
```


For Obfuscation:

```
python Obfuscate.py
```

Here we explain each argument in detail:

  * --texts: The path to the texts for obfuscation.
  * --authors_total: Total number of authors in the dataset (default 10)
  * --dir: Path to the directory containing the trained model (Contains feature set, model, etc.)
  * --trial_name: Name of the trial (human-readable) to generate the save directory (default is empty string)
  * --L: L, the number of top POS n-grams to consider for obfuscation (default 15)
  * --c: c, the length scaling constant (default 1.35)
  * --min_length: The minimum length of POS n-gram to consider for obfuscation (default 1)
  * --ig_steps: Number of steps associated with discrete integral calculation for Integrated Gradients attribution (default 1024)

  
## To Do
Update with automatic evaluation scripts