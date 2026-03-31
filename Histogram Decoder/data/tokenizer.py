"""
tokenizer.py

Manages the vocabulary for symbolic expressions in prefix notation.

The vocabulary must exactly match the operator set defined in
expression_generator.py.  Any mismatch would cause unknown tokens
at training time.

Vocabulary breakdown
---------------------
  Special (5)  : <PAD> <BOS> <EOS> <CONST> <UNK>
  Unary ops (9): sin cos exp sqrt pow2 pow3 atan asin acos
  Binary ops(4): + * - /
  Variable (1) : x
  Integers (8) : 1 2 3 4 (positive only, matching fill_leaves bias)
                 and -1 -2 -3 -4 (kept for completeness, rarely appear)
  Total        : 31 tokens

The <CONST> token is a placeholder that the decoder emits when it
predicts a real-valued constant.  The associated float value is
predicted in parallel by the regression head and refined by BFGS
post-decoding.  Integer literals (1..4) in the expression are
represented as distinct tokens with known exact values.
"""


PAD_TOKEN   = "<PAD>"
BOS_TOKEN   = "<BOS>"
EOS_TOKEN   = "<EOS>"
CONST_TOKEN = "<CONST>"
UNK_TOKEN   = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, CONST_TOKEN, UNK_TOKEN]

# Must match UNARY_OPS keys in expression_generator.py
UNARY_TOKENS = [
    "sin", "cos", "exp", "sqrt",
    "pow2", "pow3", "atan", "asin", "acos",
]

# Must match BINARY_OPS keys in expression_generator.py
BINARY_TOKENS = ["+", "*", "-", "/"]

VARIABLE_TOKENS = ["x"]

# Positive integers 1-4 (leaf constants, per _fill_leaves max_const=4)
# Negative counterparts included for model completeness
INTEGER_TOKENS = ["1", "2", "3", "4", "-1", "-2", "-3", "-4"]

ALL_TOKENS = (
    SPECIAL_TOKENS
    + UNARY_TOKENS
    + BINARY_TOKENS
    + VARIABLE_TOKENS
    + INTEGER_TOKENS
)

# All tokens that represent integer leaf nodes
_INTEGER_SET = set(INTEGER_TOKENS)


class Tokenizer:
    """
    Bidirectional mapping between token strings and integer IDs.

    Usage
    -----
    tok = Tokenizer()
    ids  = tok.encode(["sin", "x"])        # -> [5, 18]
    strs = tok.decode([5, 18])             # -> ["sin", "x"]
    full = tok.wrap(["sin", "x"])          # -> [1, 5, 18, 2]
    """

    def __init__(self):
        self.token_to_id = {t: i for i, t in enumerate(ALL_TOKENS)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad_id   = self.token_to_id[PAD_TOKEN]
        self.bos_id   = self.token_to_id[BOS_TOKEN]
        self.eos_id   = self.token_to_id[EOS_TOKEN]
        self.const_id = self.token_to_id[CONST_TOKEN]
        self.unk_id   = self.token_to_id[UNK_TOKEN]
        self.vocab_size = len(ALL_TOKENS)

    def encode(self, tokens):
        """List of token strings -> list of integer IDs."""
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids, skip_special=True):
        """
        List of integer IDs -> list of token strings.
        Skips PAD/BOS/EOS by default.
        """
        skip = {self.pad_id, self.bos_id, self.eos_id} if skip_special else set()
        return [self.id_to_token.get(i, UNK_TOKEN) for i in ids if i not in skip]

    def wrap(self, tokens):
        """
        Prepend BOS and append EOS to a token list, then encode.
        This is the target format for all decoder sequences.
        """
        return self.encode([BOS_TOKEN] + list(tokens) + [EOS_TOKEN])

    def is_const_token(self, token_id):
        """True if the given ID maps to the <CONST> placeholder."""
        return token_id == self.const_id

    def is_integer_literal(self, token_str):
        """True if the token is a known integer literal (not x, not operator)."""
        return token_str in _INTEGER_SET
