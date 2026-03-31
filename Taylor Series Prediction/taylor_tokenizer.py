PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

UNARY_TOKENS = [
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "pow2", "pow3", "pow4",
    "neg",      # unary negation, used in Mul serialisation
]
BINARY_TOKENS = ["+", "*", "/", "pow"]
VARIABLE_TOKENS = ["x"]
INTEGER_TOKENS = [str(i) for i in range(-10, 11)]   # -10 .. 10
# Rational tokens that appear as Taylor coefficients up to order 4.
# Denominators come from n! for n=1..5 and small multiples thereof.
_DENOMS = [2, 3, 4, 6, 8, 12, 24, 120]
_RATIONAL_TOKENS = []
for _den in _DENOMS:
    for _num in range(-10, 11):
        if _num == 0:
            continue
        from math import gcd as _gcd
        _g = _gcd(abs(_num), _den)
        _n2, _d2 = _num // _g, _den // _g
        if _d2 > 1:
            _RATIONAL_TOKENS.append(f"{_n2}/{_d2}")

# Remove duplicates while preserving order
_seen = set()
RATIONAL_TOKENS = []
for _t in _RATIONAL_TOKENS:
    if _t not in _seen:
        RATIONAL_TOKENS.append(_t)
        _seen.add(_t)

ALL_TOKENS = (SPECIAL_TOKENS + UNARY_TOKENS + BINARY_TOKENS + VARIABLE_TOKENS + INTEGER_TOKENS + RATIONAL_TOKENS)

class TaylorTokenizer: # Bidirectional mapping between token strings and integer IDs
    def __init__(self):
        self.token_to_id = {t: i for i, t in enumerate(ALL_TOKENS)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad_id  = self.token_to_id[PAD_TOKEN]
        self.bos_id  = self.token_to_id[BOS_TOKEN]
        self.eos_id  = self.token_to_id[EOS_TOKEN]
        self.unk_id  = self.token_to_id[UNK_TOKEN]
        self.vocab_size = len(ALL_TOKENS)

    def encode(self, tokens): # List of token strings -> list of integer IDs
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids, skip_special=True): # List of integer IDs -> list of token strings
        skip = {self.pad_id, self.bos_id, self.eos_id} if skip_special else set()
        return [self.id_to_token.get(i, UNK_TOKEN) for i in ids if i not in skip]

    def wrap(self, tokens): # Prepend BOS, append EOS, then encode.
        return self.encode([BOS_TOKEN] + list(tokens) + [EOS_TOKEN])

    def pad_sequence(self, ids, max_len): # Right-pad a sequence to max_len with pad_id
        return ids + [self.pad_id] * (max_len - len(ids))