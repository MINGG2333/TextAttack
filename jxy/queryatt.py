# Import the dataset
from textattack.datasets import HuggingFaceDataset
dataset = HuggingFaceDataset("ag_news", None, "test")
print(dataset[0])

# Import the model
import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Import the Attack
# from textattack.transformations import WordSwapEmbedding
# from textattack.search_methods import GreedyWordSwapWIR
# from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
# from textattack.goal_functions import UntargetedClassification
# from textattack import Attack
# # We'll use the greedy search with word importance ranking method again
# search_method = GreedyWordSwapWIR()
# # We're going to the `WordSwapEmbedding` transformation. Using the default settings, this
# # will try substituting words with their neighbors in the counter-fitted embedding space.
# transformation = WordSwapEmbedding(max_candidates=20)
# # We'll constrain modification of already modified indices and stopwords
# constraints = [RepeatModification(),
#                StopwordModification()]
# # Create the goal function using the model
# goal_function = UntargetedClassification(model_wrapper)
# # Now, let's make the attack from the 4 components:
# attack = Attack(goal_function, constraints, transformation, search_method)
# print(attack)
from textattack.attack_recipes import PWWSRen2019
attack = PWWSRen2019.build(model_wrapper)
# print(attack)

# Start Attack
from tqdm import tqdm # tqdm provides us a nice progress bar.
from textattack.loggers import CSVLogger # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attacker
from textattack import AttackArgs
from textattack.datasets import Dataset
attack_args = AttackArgs(num_examples=5)
attacker = Attacker(attack, dataset, attack_args)
attack_results = attacker.attack_dataset()

#The following legacy tutorial code shows how the Attack API works in detail.

#logger = CSVLogger(color_method='html')

#num_successes = 0
#i = 0
#while num_successes < 10:
    #result = next(results_iterable)
#    example, ground_truth_output = dataset[i]
#    i += 1
#    result = attack.attack(example, ground_truth_output)
#    if isinstance(result, SuccessfulAttackResult):
#        logger.log_attack_result(result)
#        num_successes += 1
#       print(f'{num_successes} of 10 successes complete.')

import pandas as pd
pd.options.display.max_colwidth = 480 # increase colum width so we can actually read the examples

logger = CSVLogger(color_method='html')

for result in attack_results:
    logger.log_attack_result(result)

from IPython.core.display import display, HTML
display(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False)))
