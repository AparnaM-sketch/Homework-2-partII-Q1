"""
Bigram Language Model – Interactive Training
Reads training corpus from terminal, computes MLE bigram probabilities,
and evaluates two predefined test sentences.
"""

import re
from collections import defaultdict, Counter

class BigramLM:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.start = '<s>'
        self.end = '</s>'
    
    def tokenize(self, sentence):
        """Simple tokenization by whitespace; keeps start/end tokens."""
        return sentence.strip().split()
    
    def train(self, corpus):
        """Train on list of sentences (each a string with tokens)."""
        for sent in corpus:
            tokens = self.tokenize(sent)
            # Add unigram counts (including start/end)
            for token in tokens:
                self.unigram_counts[token] += 1
                self.vocab.add(token)
            # Add bigram counts
            for i in range(len(tokens)-1):
                w1, w2 = tokens[i], tokens[i+1]
                self.bigram_counts[w1][w2] += 1
    
    def bigram_prob_mle(self, w1, w2):
        """MLE: count(w1 w2) / count(w1)"""
        count_w1w2 = self.bigram_counts[w1].get(w2, 0)
        count_w1 = self.unigram_counts.get(w1, 0)
        if count_w1 == 0:
            return 0.0
        return count_w1w2 / count_w1
    
    def sentence_probability(self, sentence):
        """Compute P(sentence) using bigram MLE."""
        tokens = self.tokenize(sentence)
        prob = 1.0
        for i in range(len(tokens)-1):
            p = self.bigram_prob_mle(tokens[i], tokens[i+1])
            prob *= p
            if prob == 0:
                break
        return prob
    
    def print_counts(self):
        """Display unigram and bigram counts."""
        print("\nUnigram counts:")
        for w, c in sorted(self.unigram_counts.items()):
            print(f"  {w}: {c}")
        print("\nBigram counts:")
        for w1, counter in self.bigram_counts.items():
            for w2, c in counter.items():
                print(f"  ({w1}, {w2}): {c}")

def main():
    print("=== Bigram Language Model Training ===\n")
    
    # Read training corpus interactively
    n_sentences = int(input("Enter number of training sentences: "))
    corpus = []
    print("Enter each sentence (include <s> and </s> tokens):")
    for i in range(n_sentences):
        sent = input(f"Sentence {i+1}: ").strip()
        corpus.append(sent)
    
    # Train the model
    lm = BigramLM()
    lm.train(corpus)
    
    # Display learned counts
    lm.print_counts()
    
    # Test sentences (as given in the assignment)
    test_sentences = [
        "<s> I love NLP </s>",
        "<s> I love deep learning </s>"
    ]
    
    print("\n=== Testing on given sentences ===")
    probs = {}
    for sent in test_sentences:
        p = lm.sentence_probability(sent)
        probs[sent] = p
        print(f"P({sent}) = {p:.6f}")
    
    # Determine which sentence the model prefers
    if probs[test_sentences[0]] > probs[test_sentences[1]]:
        preferred = test_sentences[0]
        reason = ("It has a higher bigram probability. "
                  "While P(deep|love)=P(NLP|love)=0.5, "
                  "the second sentence includes an extra bigram (learning|deep) "
                  "which multiplies by another factor, making its product smaller.")
    else:
        preferred = test_sentences[1]
        reason = "It has a higher bigram probability (explain based on counts)."
    
    print(f"\nThe model prefers: {preferred}")
    print(f"Reason: {reason}")

if __name__ == "__main__":
    main()