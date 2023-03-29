# Conversation_model

## Colab
Download Colab and below Model pt file.<br>
Make directory in google drive and Save these files in that directory.<br>
Then, Set the path as your directory path in each ipynb files <br>
("conversation_classifier.ipynb" and "Inference_Server.ipynb") and run codes.<br>
<br>

<strong>conversation_classifier.ipynb</strong> <br>
We trained and tested the phone scam detection model using KoBERT and FSS phone scam dataset.

<strong>Inference_Server.ipynb</strong> <br>
We implemented api server that detects phone scam for a given sentence and returns the result to the main server.

<strong>conversation_data_set.xlsx</strong> <br>
We gathered and pre-processed positive and negative phone (scam) data.

## Module

<strong>BERTClassifier.py</strong> <br>
Load BERTClassifier class

<strong>BERTClassifier.py</strong> <br>
Load BERTDataset class

<strong>main.py</strong> <br>
Predict the phone scam class and its probability for given sentence 

## Model pt file

[Download](https://drive.google.com/file/d/1jo4JT5E21U-1f10tgy1dfW6S8n9I3pDs/view?usp=share_link)
