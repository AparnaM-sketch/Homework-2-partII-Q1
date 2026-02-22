# Homework-2-partII-Q1

👩‍🎓 Student Information

      Name: MOPARTHI APARNA
      Course: CS5760 Natural Language Processing
      Department: Computer Science & Cybersecurity
      Semester: Spring 2026
      Assignment: Homework 2, Part II, Question 1

**Bigram Language Model Implementation**

  In this homework, I implemented a bigram language model from scratch using Python. The goal was to understand how n‑gram models work by building a simple one that learns from a small corpus and can compute the probability of sentences.
      
  What is a Bigram Language Model?
  
  A bigram model is a probabilistic model that predicts the next word based on the previous word only (the Markov assumption). It estimates probabilities by counting how often word pairs appear in the training data and then normalising those counts. The probability of a whole sentence is the product of the probabilities of each word given the previous one.
      
  How My Code Works
  
  Training Data Input
  
  The program first asks the user to enter the number of training sentences and then each sentence. For the given exercise, I entered the three sentences:
      
      <s> I love NLP </s>
      
      <s> I love deep learning </s>
      
      <s> deep learning is fun </s>
      
      The <s> and </s> tokens mark the start and end of sentences, which helps the model learn appropriate boundaries.
      
  Counting
  
    I defined a class BigramLM that keeps track of:
        
    1. Unigram counts – how many times each word appears.
    2. Bigram counts – how many times each pair of consecutive words appears.
       
    I used a Counter for unigrams and a defaultdict of Counters for bigrams.
      
  Estimating Probabilities (MLE)
  
    The function bigram_prob_mle(w1, w2) computes the maximum likelihood estimate:
      
      P(w2∣w1) = count(w1,w2)/count(w1)
       
      If the bigram never occurred, the probability is zero (which is a limitation, but we don't apply smoothing here to keep it simple).
      
  Sentence Probability
  
  The sentence_probability(sentence) method tokenises the input sentence (by splitting on whitespace) and then multiplies the bigram probabilities for each adjacent pair. If any probability is zero, the product becomes zero and we stop early.
      
  Testing
  
  After training, the program automatically tests the two sentences from the assignment. It prints the unigram and bigram counts, then the computed probabilities for both sentences, and finally explains which one the model prefers.
      
   Why Does the Model Prefer One Sentence?
      From the counts we get:
      
      P(I∣<s>)=2/3
      
      P(love∣I)=1 (because "I" is always followed by "love")
      
      P(NLP∣love)= 1/2(half the time "love" is followed by "NLP")
      
      P(deep∣love)= 1/2
      
      P(learning∣deep)=1 ("deep" is always followed by "learning")
      
      P(</s>∣learning)= 1/2
      
  Now for the two sentences:
      
      S1: <s> I love NLP </s>
      2/3×1×0.5×1= 0.333
      
      S2: <s> I love deep learning </s>
      2/3×1×0.5×1×0.5=0.167
      
  S1 has a higher probability because it has one fewer bigram step – the extra step in S2 (the bigram learning </s> with probability 0.5) reduces the overall product. This shows that longer sentences tend to have lower probabilities simply because more numbers are multiplied together, even if each conditional probability is reasonable.
      
  What I Learned
  
  1. Bigram models are simple but illustrate the core ideas behind language modelling: counting, normalising, and applying the chain rule.
      
  2. The zero‑probability problem: any unseen bigram makes the whole sentence probability zero – that’s why smoothing is needed in practice.
      
  3. The code is modular and can easily be extended to add smoothing, handle unknown words, or compute perplexity.
      
  Overall, this homework gave me a hands‑on understanding of how basic statistical language models work before moving on to more advanced techniques like neural language models.

📚 Overview

The program:

1. Asks the user to enter the number of training sentences and then each sentence (must include <s> and </s> tokens).

2. Trains a bigram language model on the entered corpus.

3. Displays the learned unigram counts and bigram counts.

4. Computes the probability of two test sentences using the MLE bigram probabilities:

        <s> I love NLP </s>
        
        <s> I love deep learning </s>

5. Prints which sentence the model prefers and explains why.

The implementation demonstrates the core concepts of n‑gram language modeling: counting, MLE estimation, and sentence probability computation via the chain rule.

🚀 Requirements

    Python 3.6 or higher
    No external libraries required (uses only collections and re from the standard library)

📝 How to Run

    Clone the repository or download part2_q1.py.
    Open a terminal in the directory containing the script.
    Run the command:
    
    bash
    python part2_q1.py
    Follow the prompts to enter the training corpus.
    For the example in the assignment, you would enter:
    
    text
    Enter number of training sentences: 3
    Enter each sentence (include <s> and </s> tokens):
    Sentence 1: <s> I love NLP </s>
    Sentence 2: <s> I love deep learning </s>
    Sentence 3: <s> deep learning is fun </s>
    The program will then display the counts and the results for the test sentences.

📊 Example Output

    === Bigram Language Model Training ===
    
    Enter number of training sentences: 3
    Enter each sentence (include <s> and </s> tokens):
    Sentence 1: <s> I love NLP </s>
    Sentence 2: <s> I love deep learning </s>
    Sentence 3: <s> deep learning is fun </s>
    
    Unigram counts:
      </s>: 3
      <s>: 3
      I: 2
      NLP: 1
      deep: 2
      fun: 1
      is: 1
      learning: 2
      love: 2
    
    Bigram counts:
      (<s>, I): 2
      (<s>, deep): 1
      (I, love): 2
      (love, NLP): 1
      (love, deep): 1
      (NLP, </s>): 1
      (deep, learning): 2
      (learning, </s>): 1
      (learning, is): 1
      (is, fun): 1
      (fun, </s>): 1
    
    === Testing on given sentences ===
    P(<s> I love NLP </s>) = 0.333333
    P(<s> I love deep learning </s>) = 0.166667
    
    The model prefers: <s> I love NLP </s>
    Reason: It has a higher bigram probability. While P(deep|love)=P(NLP|love)=0.5, the second sentence includes an extra bigram (learning|deep) which multiplies by another factor, making its product smaller.


🔧 Code Structure

  Class BigramLM
  
      __init__() – Initialises counters and vocabulary.
      
      tokenize(sentence) – Splits a sentence into tokens (by whitespace).
      
      train(corpus) – Updates unigram and bigram counts from a list of sentences.
      
      bigram_prob_mle(w1, w2) – Returns the MLE probability of w2 given w1.
      
      sentence_probability(sentence) – Computes the joint probability of a sentence using the chain rule and bigram probabilities.
      
      print_counts() – Displays the unigram and bigram counts in a readable format.
      
      Function main()
      
      Handles user input for the training corpus.
      
      Trains the model.
      
      Evaluates the two fixed test sentences.
      
      Compares probabilities and prints the preferred sentence with a reason.

📈 Explanation of the Result

The bigram probabilities learned from the corpus:

      P(I∣<s>)= 2/3
      
      P(love∣I)=2/2=1
      
      P(NLP∣love)=1/2=0.5
      
      P(deep∣love)=1/2=0.5
      
      P(learning∣deep)=2/2=1
      
      P(</s>∣learning)=1/2=0.5
      
      For <s> I love NLP </s>: 2/3×1×0.5×1= 0.3333
      
      For <s> I love deep learning </s>: 2/3×1×0.5×1×0.5= 0.1667

The first sentence is more probable because it has fewer bigram steps, and the extra step in the second sentence reduces the overall probability.
