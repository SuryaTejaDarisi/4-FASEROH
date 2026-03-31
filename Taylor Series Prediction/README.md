# Taylor Series Prediction
For a given function, the objective is to generate Taylor Series Expansion of the function up to order 4. Sequence-to-Sequence learning models such as LSTM and Transformer are implemented from scratch to predict the terms of Taylor series of various mathematical functions. Hence, the objective of this problem statement can be framed as a translation problem between two symbolic sequences.


Firstly, install all the required packages from requirements.txt
```
pip install -r .\requirements.txt
```
-------------------------------------------

Next we have two options to train i.e., training a LSTM or a Transformer.
At each step we've options to save the data generated or generate freshly and use them.
Early Stopping is used in the Training process.

Train LSTM:
```
    python taylor.train_taylor --model lstm --n_samples 5000 --save_data --epochs 30
```



Train Transformer + Load saved dataset:
```
    python taylor.train_taylor --model transformer --data_path .\data\dataset.json --epochs 30
```

To evaluate Trained Models: (By default, if both models are provided, then they are compared)
```
python evaluate.py --data_path ".\data\test.json"
```

To plot the loss curves across Training and Validation:
```
python plot_taylor.py
```
