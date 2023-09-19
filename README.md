# Text Summarization Using TF-IDF and Transformers

In this project, we explore text summarization techniques using TF-IDF and Transformers, two different approaches for generating concise and informative summaries from text data. We also evaluate the quality of the generated summaries using ROUGE scores and then deploy the transformer based model on huggingface spaces using Streamlit .

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Downloading the Dataset](#downloading-the-dataset)
  - [Preprocessing the Dataset](#preprocessing-the-dataset)
  - [Exploring the Dataset](#exploring-the-dataset)
- [Text Summarization](#text-summarization)
  - [TF-IDF Summarization](#tf-idf-summarization)
  - [Summarization Using Transformers](#summarization-using-transformers)
- [Evaluation](#evaluation)
  - [ROUGE Score Evaluation](#rouge-score-evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Text summarization is the process of condensing a large amount of text into a shorter, coherent version while preserving its key information. In this project, we explore two popular approaches for text summarization: TF-IDF and Transformers. TF-IDF relies on statistical analysis, while Transformers leverage pre-trained deep learning models.

## Getting Started

### Setting Up the Environment

To get started, set up your Python environment. You can use Google Colab or your local machine. Install the required libraries and packages. To facilitate dataset access, set up your Kaggle API token and mount Google Drive if necessary.

### Downloading the Dataset

We used the "newspaper-text-summarization-cnn-dailymail" dataset from Kaggle for our project. You can download it using the Kaggle API or obtain a similar dataset of your choice.

### Preprocessing the Dataset

Before diving into text summarization, perform some data preprocessing. Load the dataset into a Pandas DataFrame, handle any missing values, and prepare the data for summarization.

### Exploring the Dataset

Examine the structure of the dataset. Our dataset contains 'article' (full text of news articles) and 'highlights' (corresponding article summaries) columns.

## Text Summarization

### TF-IDF Summarization

TF-IDF (Term Frequency-Inverse Document Frequency) is a traditional text summarization technique. We implemented TF-IDF using Scikit-Learn's `TfidfVectorizer`. Stop words and tokenization were applied for text cleaning. A custom function calculated sentence scores based on TF-IDF values, and summaries were generated.

### Summarization Using Transformers

Transformers, particularly the "facebook/bart-large-cnn" model, provide state-of-the-art text summarization capabilities. We used the Hugging Face Transformers library to create a custom summarization pipeline, generating summaries for news articles.

## Evaluation

### ROUGE Score Evaluation

To evaluate the quality of our generated summaries, we used the ROUGE scoring system. ROUGE calculates metrics (ROUGE-1, ROUGE-2, ROUGE-L) to measure the overlap between the generated summary and the reference summary. We loaded the ROUGE scorer and computed ROUGE scores for each generated summary.

## Streamlit App for Text Summarization

We have also developed a Streamlit web application that allows you to generate text summaries interactively. The app uses the "facebook/bart-large-cnn" model from Hugging Face Transformers for summarization and evaluates the summaries using the ROUGE scoring system.

### Running the Streamlit App

To run the Streamlit app locally, follow these steps:

1. Make sure you have Python installed on your system.

2. Install the required Python libraries by running the following command in your project directory:

   ```bash
   pip install streamlit transformers rouge-score

## Results

We present the results of our text summarization experiments, including generated summaries and their associated ROUGE scores, in the project's notebook. The project is also deployed in huggingface spaces .
* Link : https://huggingface.co/spaces/crockrocks/text-summarization

## Contributing

If you're interested in contributing to this project, feel free to open issues, create pull requests, or contact us for collaboration.

