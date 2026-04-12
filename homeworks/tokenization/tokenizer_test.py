import pytest

from collections import Counter

from tokenizer import BPETokenizer, DEFAULT_UNK, DEFAULT_PAD


@pytest.fixture
def simple_tokenizer():
    return BPETokenizer(vocab_size=10)

@pytest.fixture
def fitted_tokenizer():
    tokenizer = BPETokenizer(vocab_size=150)
    corpus = ["hello world", "hello there", "world peace"]
    tokenizer.fit(corpus)
    return tokenizer

def test_init():
    tokenizer = BPETokenizer(vocab_size=100)
    assert tokenizer.vocab_size == 100
    assert tokenizer.unk == DEFAULT_UNK
    assert tokenizer.pad == DEFAULT_PAD

def test_init_invalid_vocab_size():
    with pytest.raises(ValueError):
        BPETokenizer(vocab_size=0)

def test_fit(simple_tokenizer):
    corpus = ["hello"]
    vocab = simple_tokenizer.fit(corpus)
    assert isinstance(vocab, Counter)
    assert DEFAULT_UNK in vocab
    assert DEFAULT_PAD in vocab
    assert 'h' in vocab
    assert 'e' in vocab

def test_tokenize_without_fit(simple_tokenizer):
    with pytest.raises(ValueError):
        simple_tokenizer.tokenize("hello")

def test_tokenize(fitted_tokenizer):
    tokens = fitted_tokenizer.tokenize("hello")
    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)
    assert all(token in fitted_tokenizer.vocab or token == DEFAULT_UNK for token in tokens)

def test_detokenize(fitted_tokenizer):
    tokens = fitted_tokenizer.tokenize("hello")
    text = fitted_tokenizer.detokenize(tokens)
    assert isinstance(text, str)
    assert text == "hello"

def test_encode_decode(fitted_tokenizer):
    original = "hello world"
    encoded = fitted_tokenizer.encode(original)
    assert isinstance(encoded, list)
    assert all(isinstance(id, int) for id in encoded)

    decoded = fitted_tokenizer.decode(encoded)
    assert decoded == original

def test_unknown_tokens(fitted_tokenizer):
    tokens = fitted_tokenizer.tokenize("xyz123")  # Characters not in training data
    assert DEFAULT_UNK in tokens

def test_empty_input(fitted_tokenizer):
    assert fitted_tokenizer.tokenize("") == []
    assert fitted_tokenizer.detokenize([]) == ""
    assert fitted_tokenizer.encode("") == []
    assert fitted_tokenizer.decode([]) == ""

def test_lorem_ipsum_tokenize():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    tokenizer = BPETokenizer(vocab_size=1000)
    corpus = [ text ] * 1000
    tokenizer.fit(corpus)
    tokens = tokenizer.tokenize("Lorem ipsum dolor sit amet")
    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)
    assert all(token in tokenizer.vocab or token == DEFAULT_UNK for token in tokens)

def test_fitted_tokenizer_vocab(fitted_tokenizer):

    assert "hello " in fitted_tokenizer.vocab
    assert "world" in fitted_tokenizer.vocab