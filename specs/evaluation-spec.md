# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
[blank — write out the accuracy formula in plain English.
 What counts as "correct"? What do you divide by?]
```
The accuracy formula for the classification of all the episodes should just be the correct predictions (predictions that match ground_truth) divided by total number of episodes. For example, if we have 10 right predictions by the classifier out of 20 episodes, the accuracy formula should be 10/20, which is 50%.

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. ...
 2. ...
 3. ...]
```
1. Normalize the labels in 'predictions' and 'ground_truth'. That way, when we compare the labels in 'prediction' and 'ground_truth', we are able to see comparisons side-by-side without worrying about whitespaces, uppercases, etc.
2. Compute the matches in 'predictions' and 'ground_truth'. For each label in 'predictions' that matches the label in 'ground_truth', we have 1 correct prediction.
3. Once we have the correct prediction count, we divide this number by the total number of labels. In this case, it would be the length of 'predictions' or 'ground_truth'.

---

**Edge case — what if both lists are empty?**

```
[blank — what should the function return? Why?]
```
If both inputs are empty, then we simply return accuracy as 0. Because we have no list to work with, we cannot compute the accuracy, and therefore should return either a 0 or null.

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

[blank — what does compute_accuracy() return for these inputs? Show your work.]
```
Because 'predictions' and 'ground_truth' are in the same order, we just count the labels that are the same in both orders. In this example,
1. interview, interview (+1)
2. solo, solo (+1)
3. panel, solo
4. interview, narrative

overall, because we have 1 and 2 with the same label in the same order, we have 2 correct labels. We then divide this by the total number of labels, which in this case is 4 (prediction + ground_truth length = 4). Therefore, we have 2/4, which is 50%.
---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
[blank — be precise. When does an episode count as correctly classified
 for the "interview" class, for example?]
```
For an episode to be classified as correctly labeled, we calculate the number of 'labels' that we have, and see how many times it was correctly labeled.
For example, if we have 20 episodes, and 5 of them are labeled 'interview', we want to see how many of the 5 'interview' were correctly labeled. If 3 of the 'interview' were actually labeled 'interview', then we would have 3/5, which would be 60%.

---

**What does "total" mean for a given class?**

```
[blank — is "total" the total number of predictions, or something more specific?]
```
Total is the total number of 'label' in the list. If we have 20 episodes, and 4 of them are labeled 'solo', then our total for the given class 'solo' would be 4. 
---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. Initialize ...
 2. Loop over ...
 3. For each pair (predicted, truth) ...
 4. After the loop ...
 5. Return ...]
```

1. Initialize a dictionary with 4 keys (for the 4 labels), each containing a dictionary with keys 'correct', 'total' and 'accuracy' of type int, int and float respectively. 'correct' and 'total' should be initialized to 0.
2. Loop over each value in 'ground_truth'. For each value in 'ground_truth', we increment the 'total' in that 'label' by 1, and compare it to the value in 'predictions'. If the prediction is right, we increment 'correct' by 1.
3. After we finish the loop, we create another loop, this time to calculate the 'accuracy' in the 'labels' dictionary. For each label that we have, we calculate 'correct'/'total', and store that as a float in 'accuracy'.
4. We return the dictionary.

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
[blank — what should accuracy be set to? Why?
 Hint: look at the docstring in evaluate.py.]
```

If a class has no example in ground_truth, the 'accuracy' should be set to 0.0.
---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

[blank — fill in the per-class results table below]

label       correct  total  accuracy
----------  -------  -----  --------
interview   [1]      [1]      [1.0]
solo        [1]      [2]      [0.5]
panel       [1]      [1]      [1.0]
narrative   [0]      [1]      [0.0]
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
