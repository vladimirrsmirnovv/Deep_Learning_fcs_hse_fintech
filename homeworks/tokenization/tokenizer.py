from typing import List
from collections import Counter

import re

DEFAULT_UNK = '[UNK]'
DEFAULT_PAD = '[PAD]'


class BPETokenizer():
    """
    Byte Pair Encoding (BPE) tokenizer implementation.

    BPE is a subword tokenization algorithm that iteratively merges the most frequent
    pairs of characters or character sequences to build a vocabulary of subwords.
    """

    def __init__(
            self,
            vocab_size=8192,
            # percent_bpe=0.2, # todo should we support it?
            UNK=DEFAULT_UNK,
            PAD=DEFAULT_PAD,
        ):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab_size: Maximum size of the vocabulary
            UNK: Token to use for unknown tokens
            PAD: Token to use for padding

        Raises:
            ValueError: If vocab_size is less than 1
        """

        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.vocab_size = vocab_size

        self.unk = UNK
        self.pad = PAD

        self.vocab = Counter()
        self.merges = {}
        self._token_to_id = None
        self._id_to_token = None

    def _get_stats(self, subwords: List[List[str]]) -> Counter:
        """
        Count frequency of adjacent token pairs in all subwords.

        Args:
            subwords: List of lists, where each inner list contains tokens

        Returns:
            Counter with pairs of adjacent tokens as keys and their frequencies as values
        """
        pairs = Counter()

        for word in subwords:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1

        return pairs

    def _perform_merge(
        self,
        pairs: Counter,
        subwords: List[List[str]],
        new_token: str,
        best_pair: tuple[str, str]
    ) -> List[List[str]]:
        """
        Merge the most frequent pair in all subwords and update pair statistics.

        Args:
            pairs: Counter with token pair frequencies
            subwords: List of lists of tokens to update
            new_token: The merged token to insert
            best_pair: The pair of tokens to merge

        Returns:
            Tuple of (updated pairs counter, updated subwords list)
        """

        # Easy way: rebuild pair stats after merge
        new_pairs = Counter()
        new_subwords = []

        a, b = best_pair

        for word in subwords:
            merged_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged_word.append(new_token)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            new_subwords.append(merged_word)

        new_pairs = self._get_stats(new_subwords)

        return new_pairs, new_subwords

    def fit(self, text_corpus: List[str]) -> Counter:
        """
        Train the tokenizer on a corpus of text.

        This method implements the BPE algorithm:
        1. Start with a vocabulary of individual characters
        2. Count frequencies of adjacent character pairs
        3. Merge the most frequent pair and add to vocabulary
        4. Repeat until reaching target vocabulary size

        Args:
            text_corpus: List of text strings to train on

        Returns:
            Counter object containing the vocabulary with frequencies
        """
        self.vocab = Counter()
        self.merges = {}
        self._token_to_id = None
        self._id_to_token = None

        # Initialize vocabulary with characters, not bytes
        subwords = []
        for text in text_corpus:
            subword = list(text)
            subwords.append(subword)
            self.vocab.update(subword)

        # Add special tokens
        self.vocab[self.unk] = 1
        self.vocab[self.pad] = 1

        # Calculate how many merges we need to reach target vocabulary size
        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            return self.vocab

        # Calculate initial statistics
        pairs = self._get_stats(subwords)

        # Perform merges until we reach target vocabulary size
        for i in range(num_merges):
            if not pairs:
                break

            best_pair, freq = pairs.most_common(1)[0]
            if freq <= 0:
                break

            new_token = ''.join(best_pair)

            # Save merge rank: smaller rank = earlier merge = higher priority
            self.merges[best_pair] = i
            self.vocab[new_token] = freq

            pairs, subwords = self._perform_merge(
                pairs=pairs,
                subwords=subwords,
                new_token=new_token,
                best_pair=best_pair,
            )

        return self.vocab

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subwords based on the vocabulary.

        This applies the learned BPE merges to the input text:
        1. Start with individual characters
        2. Iteratively apply merges in the order they were learned
        3. Replace unknown tokens with UNK token

        Args:
            text: Input text string

        Returns:
            List of subword tokens

        Raises:
            ValueError: If tokenizer hasn't been fitted yet
        """
        if not self.merges:
            raise ValueError("Tokenizer must be fitted before tokenization")

        if text == "":
            return []

        # Initialize with individual characters
        subwords = list(text)

        while True:
            if len(subwords) < 2:
                break

            # Calculate current adjacent pairs
            pairs = self._get_stats([subwords])

            # Find the best pair to merge based on learned merge order
            best_pair = None
            best_rank = float('inf')

            for pair in pairs:
                if pair in self.merges and self.merges[pair] < best_rank:
                    best_rank = self.merges[pair]
                    best_pair = pair

            # If no valid pair found, stop
            if best_pair is None:
                break

            new_token = ''.join(best_pair)
            _, merged_subwords = self._perform_merge(
                pairs=pairs,
                subwords=[subwords],
                new_token=new_token,
                best_pair=best_pair,
            )
            subwords = merged_subwords[0]

        # Replace unknown tokens with UNK
        return [t if t in self.vocab else self.unk for t in subwords]

    def detokenize(self, subwords: List[str]) -> str:
        """
        Join tokens and remove UNK tokens.

        Args:
            subwords: List of tokens to join

        Returns:
            Reconstructed text string
        """
        return ''.join(token for token in subwords if token != self.unk)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token ids.

        Args:
            text: Input text string

        Returns:
            List of token ids
        """
        if self._token_to_id is None:
            # Lazily initialize token-to-id mapping
            self._token_to_id = {token: i for i, token in enumerate(self.vocab)}

        tokens = self.tokenize(text)
        return [self._token_to_id.get(token, self._token_to_id[self.unk]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token ids back to text.

        Args:
            token_ids: List of token ids

        Returns:
            Reconstructed text string
        """
        if self._id_to_token is None:
            # Lazily initialize id-to-token mapping
            self._id_to_token = {i: token for i, token in enumerate(self.vocab)}

        tokens = [self._id_to_token.get(id, self.unk) for id in token_ids]
        return self.detokenize(tokens)
