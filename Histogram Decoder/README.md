# Symbolic Regression of Distributions via Seq-to-Seq Transformers

Imagine you're a physicist running an experiment. You shoot a million particles at a target
and measure how many particles land in each region of a detector. The result is a histogram:
a bar chart where each bar represents a "bin" and its height is the count of particles that
fell there.

Now here is the question: **what mathematical function describes the shape of that histogram?**

A physicist might manually guess and check: "does this look like a Gaussian? A power law?
Maybe a combination of sine and exponential?" This process is slow, subjective, and only
works if the physicist already knows what family of functions to try.

This project trains a neural network to do that job automatically. Given any histogram,
the model reads it and writes out the mathematical formula that generated it. For example,
it reads a histogram shaped like a skewed bell and outputs `sin(x) / (exp(x) + 1)`.

This is framed as a **translation problem**: just like Google Translate converts English
sentences into French sentences, this model converts "histogram language" into "math language".
