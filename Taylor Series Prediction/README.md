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
