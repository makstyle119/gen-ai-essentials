# GenAi Essentials

this is my journey to learn and understand GenAi essentials.

## Questions & Answers

**Q1- What Is AI - (Artificial Intelligence) ?** <br />
**A1-** Machine that perform jobs that mimics human behavior.

this include:
- problem-solving
- decision-making
- understanding natural language
- recognizing speech and images

**Note:** Al's Goal is to interpret, analyze and response to human actions.To simulate human intelligence in machine.

**Q2- What Is ML - (Machine Learning) ?** <br />
**A2-** Machine that get better at a task without explicit programming. <br />
there are few types of machine learning:
- Types of Machine Learning
    - Learning Problem
        - Supervised Learning
            - using a model to learn a mapping between input examples and the target output.
            - **usage**
                - when labels are available and you want precise outcome.
                - when you need a specific value return.
        - Unsupervised Learning
            - using a model to describe or extract relationships in data.
            - **usage**
                - when labels are not available and you don't want precise outcome.
                - when you are trying to make sense of data.
        - Reinforcement Learning
            - an agent operates in an environment and must learn to operate using feedback.
            - **usage**
                - when you want to simulate human behavior.
                    - Game AI
                    - Learning Tasks
                    - Robot Navigation
            <!-- - **note:** you need to provide reward for the agent to learn. -->
    - Hybrid Learning
        - Semi-Supervised Learning
            - training a model to learn from both labeled and unlabeled data.
        - Self-Supervised Learning
            - farmed as supervised learning problem in order to apply supervised learning algorithms.
        - Multi-Instance Learning
            - individual examples are unrelated, instead, bags or group of samples are labeled.
    - Statistical Interface
        - Inductive Learning
            - using evidence to determine the outcome.
        - Deductive Learning
            - using general rules to determine the outcome.
        - Transductive Learning
            - using statistical learning theory to determine the outcome.
    - Learning Techniques
        - Multi-Task
            - fitting a single model to multiple tasks.
        - Transfer Learning
            - learning from a model trained on related tasks.
        - Active
            - model able to query human like in learning process.
        - Online
            - learning while data is being collected.
        - Esemble
            - multiple models trained together.
- Division Of Machine Learning
    - Classified Machine Learning
        - Supervised Machine Learning
            - here we have data with labels.
                - here we will first provide label data then provide unlabeled data to model can label it.
            - we use 2 method to do supervised learning:
                - Classification
                    - classification is a process of finding a function to divide a dataset into classes/categories.
                        - **Classification Algorithm**
                            - Logistic Regression
                            - K-Nearest Neighbors (KNN)
                            - Support Vector Machines (SVM)
                            - Kernel SVM
                            - Naive Bayes
                            - Decision Trees Classification
                            - Random Forest Classification
                - Regression
                    - Regression is a process of finding a function to correlate a dataset into continuous values.
                        - **Regression Algorithm**
                            - Simple Liner Regression
                            - Multiple Liner Regression
                            - Polynomial Regression
                            - Support Vector Regression (SVR)
                            - Decision Tree Regression
                            - Random Forest Regression
        - Unsupervised Machine Learning
            - here we have data without labels.
            - We use 2 method to do unsupervised learning:
                - Clustering
                    - Clustering is a process of grouping similar unlabeled data points together.
                        - **Clustering Algorithm**
                            - K-Means
                            - DBScan
                            - K-Medoids
                - Association
                    - Association is a process of finding patterns in data.
                        - **Association Algorithm**
                            - Apriori
                            - Euclat
                            - FP-Growth
                - Dimensionality Reduction
                    - Dimensionality Reduction is a process of reducing the number of features in a dataset.
                        - **Dimensionality Reduction Algorithm**
                            - Principal Component Analysis (PCA)
                            - Linear Discriminant Analysis (LDA)
                            - Generalized Discriminant Analysis (GDA)
                            - Singular Value Decomposition (SVD)
                            - Latent Decrease Analysis (LDA)
                            - Latent Semantic Analysis (LSA, pLSA, GLSA)
                            - t-SNE
        - **Note:** Supervised Learning is more accurate than Unsupervised Learning, also unsupervised learning need human validation to validate results.
    - Reinforcement Machine Learning
        - Real Time Decision
        - Game AI
        - Learning Tasks
        - Robot Navigation
    - Ensemble Machine Learning
        - Bagging 
        - Boosting
        - Stacking
    - Neural Network And Deep Learning
        - Neural Network (NN)
            - often describe as mimicking the human brain, a neuron/node represents an algorithm for processing information. Data flow through the network. and based on the input, the output is calculated. and may have multiple layers.
            - if any neural network have 3 or more then 3 hidden layers it's deep learning otherwise simple neural network.
            - **note:** Deep Learning is a subset of Machine Learning.
            - Neural Network have few things:
                - Feed Forward Neural Network (FNN)
                    - always move forward.
                - backpropagation (BP)
                    - move backward to adjust the weight to improve the outcome. this is how a neural network learn.
                - Loss Function
                    - function to compare the predicted value with the actual value. (how bad is the prediction)
                - Activation Function
                    - function to calculate the predicted value. (how good is the prediction)
                - Dense
                    - layer of neurons increase.
                - Sparse
                    - layer of neurons decrease.
            - In Deep Learning, we use:
                - Supervised Learning
                    - Fully-connected Feed Forward (FNN)
                    - Recurrent Neural Network (RNN)
                    - Convolutional Neural Network (CNN)
                - Unsupervised Learning
                    - Deep Belief Network (DBN)
                    - Stacked Auto Encoders (SAE)
                    - Restricted Boltzmann Machines (RBMs)
        - Convolutional Neural Network (CNN)
        - Recurrent Neural Network (RNN)

**Q3- What Is DL - (Deep Intelligence) ?** <br />
**A3-** Means a Machine which can perform tasks like human / mimic human behavior.

**Q4- What Is GenAI ?** <br />
**A4- Generative AI** is a specialized subset of AI that generates out comes eg: images, text, audio, video, etc.

It often involves advanced machine learning techniques:
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Transformers models eg GPT

|      Header    |           AI - (Artificial Intelligence)              |                 GenAI - (Generative AI)                |
| -------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| functionality  | AI focus on understanding and decision-making         | GenAI focus on creating new things                     |
| Data Handling  | AI analyzes and makes decision based on existing data | GenAI use existing data to generate new, unseen output |
| Applications   | Spans across various sectors, including data analysis, automation, natural language processing (NLP) and healthcare | Create and innovate, focus on content creation, synthetic data generation, deepfakes, and |

**Q5- What Is NLP - (Natural Language Processing) ?** <br />
**A5- NLP - (Natural Language Processing)** is Machine Learning technique that enables computers to process and understand human language.

**Q6- What Is Regression ?** <br />
**A6- Regression** is the process of predicting continuous values. imagine you need to predict the price of a house based on its size and location. Regression is a type of machine learning model that is used to predict continuous values. it will estimate the price of a house based on its size and location by learning patterns from the available data. like price of house in same location will be similar to each other. and also price of house in same size will be similar to each other. with this model, you can predict the price of a house based on its size and location. <br />
there are various algorithms for regression.
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**Q7- What Is vector and error in regression ?** <br />
**A7-** **vector** is a line that connects two points. **error** is the distance between the actual value and the predicted value.

**Q8- What is Classification ?** <br />
**A8- Classification** is a process of finding a function to divide a labeled dataset into classes/categories. <br />
there are few classification algorithms:
- Logistic Regression
- Decision Trees/Random Forest
- Neural Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)

**Q9- What is Clustering ?** <br />
**A9- Clustering** is a process of grouping similar unlabeled data points together. <br />
there are few clustering algorithms:
- K-Means
- K-Medoids
- Density Based
- Hierarchical Clustering

**Q9- What is BERT - (Bidirectional Encoder Representations from Transformers) ?** <br />
**A9- BERT - (Bidirectional Encoder Representations from Transformers)** model released by google in 2018
- you have take Transformer Architecture and just take the encoder and stack them you would get BERT.
- BERT is bi-direction language means it can read from left to right and right to left.
    - which means BERT (neuron network understand the context all over at once not like first thing first or last thing first)
- BERT is encoder only that's mean you input a lot of text and a little will come out.
- BERT is pre-trained on:
    - Masked Language Model (MLM)
        - provide input and predict the masked word.
        - think of asking BERT to fill in the blanks in a sentence.
    - Next Sentence Prediction (NSP)
        - BERT is provided two sentence A & B.
        - BERT will predict if sentence B is next to sentence A.
- BERT can then be fine-tuned to perform various other tasks.
    - Named Entity Recognition (NER)
    - Question Answering (QA)
    - Sentence Pair Tasks
    - Summarization
    - Feature Extraction / Embedding
- BERT is available in multiple models: 
    - BERT Base 100M Parameters
    - BERT Large 240M Parameters
    - BERT Tiny 4M Parameters
    - and ~24 other models
- BERT variants:
    - roBERTa
    - DistiBERT
    - ALBERT
    - ELCTRA
    - DeBERTa
- BERT is the baseline in NLP.

**Q10- What is Sentence Transformers ?** <br />
**A10- Sentence Transformers aka SBERT** is a build on top of BERT. It create a single vector for an entire sentence. when comparing similar sentence this much more performance than simply using BERT which looks at every word.

**Q10- What is Perceptron ?** <br />
**A10- Perceptron** is an algorithm for supervised learning of binary classifiers invented in 1943 and machine build in 1957
- The Mark1 perceptron was design for image recognition.
- A Perceptron is a single neuron model that was a precursor to larger neural networks.
- **Basic Perceptron Network**
    - A basic Perceptron has an input layer and output layer 
    - Each layer contains a number of nodes
    - Nodes between layers have established connections that are weighted
        - The amount of nodes in the Input layer is determine by:
            - The number of dimensions for the inputted vectors
            - The input data is a collection of vectors.
                - eg: A vector that has an X and Y would have 2 input nodes.
            - The input layer is just connections points, it does not modify the data.
        - The amount of nodes in the Output layer is determine by:
            - The application of neural net
                - Yes/No Classification would only have one output node.
                    - (would not matters if there were 1000s of input nodes)
        - The output nodes (and other layers) can modify (compute) new values based on inputted data.

## Papers:
[ChinChilla](https://arxiv.org/pdf/2203.15556)