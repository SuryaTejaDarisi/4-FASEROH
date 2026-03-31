<<<<<<< HEAD
'''
pip install -r .\requirements.txt
'''

Train LSTM:
'''
    python -m taylor.train_taylor --model lstm --n_samples 5000 --save_data --epochs 30
'''

Train Transformer + Load saved dataset:
'''
    python -m taylor.train_taylor --model transformer --data_path .\data\dataset.json --epochs 30
'''
=======
```
pip install -r .\requirements.txt
```

Train LSTM:
```
    python -m taylor.train_taylor --model lstm --n_samples 5000 --save_data --epochs 30
```

Train Transformer + Load saved dataset:
```
    python -m taylor.train_taylor --model transformer --data_path .\data\dataset.json --epochs 30
```
>>>>>>> 5e6fe2287edb9df434b1c6b59525a05c55562654
