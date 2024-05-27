# Introduction
The internet fosters global communication but also enables cyberbullying.
Cyberbullying, involving harassment on social media, poses severe psychological risks.
Despite awareness, many users experience or are unaware of cyberbullying.
# Types and Impact of Cyberbullying
Cyberbullying includes racism, sexism, and cyber aggression.
Motivated by hate based on race, nationality, religion, gender, etc.
Affects people of all ages, leading to issues like low self-esteem, anxiety, and even suicide.
# Prevalence and Challenges
Massive volume of social media posts contains offensive content.
25% of internet users and 1 in 3 teenagers face online sthreats.
Stopping cyberbullying is difficult but essential.
# Goal
 To build a model that identifies aggression in text data

# The task of aggression identification involves classifying text into one of three predefined categories
## Overtly Aggressive (OAG): 
  This class includes texts that exhibit explicit aggression. These texts are likely to contain clear and direct aggressive language, threats, or abusive terms.
## Covertly Aggressive (CAG): 
  This class includes texts that exhibit implicit aggression. These texts may contain subtle, indirect, or sarcastic language that conveys aggression in a less obvious manner compared to overt aggression.
## Non-aggressive (NAG): 
  This class includes texts that do not exhibit any form of aggression. These texts are neutral and do not contain any aggressive language or undertones.

# Tokenizer
The tokenizer used with BigBird is similar to those used with other Transformer models, such as BERT, but adapted for handling longer sequences efficiently.
## Tokenization Process:
The tokenizer splits text into tokens, which are subword units that the model can process.
It typically uses a vocabulary trained with methods like WordPiece (used by BERT) or SentencePiece (used by T5).
## Special Tokens:
Special tokens like [CLS], [SEP], [MASK], and others are used to signify the start of the sequence, separation between sequences, masked tokens for masked language modeling, etc.
## Handling Long Sequences:
The tokenizer can handle long sequences by splitting them into manageable chunks that fit within the model's maximum sequence length while preserving context through attention mechanisms.
Unlike traditional Transformer models that have quadratic complexity with respect to the input sequence length due to full self-attention, BigBird introduces sparse attention mechanisms, making it linear in complexity.
# BigBird Model
## Architecture:
  BigBird uses a combination of three types of attention:
  # Global attention: 
  Certain tokens (like [CLS] in BERT) attend to all other tokens.
  # Random attention: 
  Tokens attend to a fixed number of randomly selected other tokens.
# Sliding window (local) attention: 
  Tokens attend to a fixed-size window of neighboring tokens.
These sparse attention mechanisms allow BigBird to capture long-range dependencies while maintaining computational efficiency.
# Applications:
  Due to its ability to process long sequences, BigBird is particularly useful for tasks involving long documents such as question answering, document classification, summarization, and more.

# Where can be used
Social networks (e.g. Instagram, facebook,Twitter …)
Online markets (e.g. list.am, Wildberries, Ozon …)
And anywhere where you can write your comment or description.


