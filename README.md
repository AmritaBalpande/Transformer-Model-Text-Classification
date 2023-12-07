# Transformer-Model-Text-Classification
Transformers, with their powerful self-attention mechanism, have revolutionized the field of natural language processing (NLP). They excel at various tasks, including text classification, where the goal is to assign a predefined category or label to a piece of text.This report details the development and evaluation of a simple transformer-based model for text Classification on the IMDb movie review dataset uing BERT.

Steps to Run the Code:

You need to set up a new environment on your computer. However, it is not compulsory to use your local machine, you can train a model on, let's say Google Colab and download the trained model to server the requests for classification.

Install required packages
Once the virtual environment is activated run the following command to get the required packages...

pip install transformers[torch] pandas datasets pyarrow scikit-learn

Get the dataset
Here, I am using a dataset downloaded from Kaggle, it is a IMDB movie review dataset, and it has around 50K samples with two labels positive and negative, but this can be implemented to more than two classes as well.

Create a dataset for BERT

Read the dataset
Now, Read the dataset

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/IMDB Dataset.csv')
df.head()
You will see that there are two columns, review and sentiment, the problem we have is a binary classification of the column review.

Process the dataset

The data should be formatted before it is passed to the Bert base uncased, this model requires input_ids, token_type_ids, attention_mask, label, and text, also there is a particular way we need to pass the data and that is why we have installed pyarrow and datasets.

To tokenize the text, we will use AutoTokenizer. The following piece of code will initialize the tokenizer.

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


Now, process the data in the CSV file, for this I will write a function, in this function the tokenizer uses max_length=128, you can increase that, but since I am just showing the workings, I will use 128.
from typing import Dict

    def process_data(row) -> Dict:
        # Clean the text
        text = row['review']
        text = str(text)
        text = ' '.join(text.split())
        # Get tokens
        encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # Convert string to integers
        label = 0
        if row['sentiment'] == 'positive':
            label += 1

        encodings['label'] = label
        encodings['text'] = text

        return encodings

  Check the working of the function.
  
  print(process_data({'review': 'this is a sample review of a movie.','sentiment': 'positive'}))

  Pass each row of the dataset and process the review and convert the sentiment into int since the dataset consists of 15K samples, but as I said, for demonstration purposes, I will only use 1K samples.
  processed_data = []

  for i in range(len(df[:1000])):
    processed_data.append(process_data(df.iloc[i]))

  Generate the dataset
  Generate the dataset in a format required by the Trainer module of the transformers library.
  The code piece below converts the list of encodings into a data frame and split that into a training and validation set of data.
  
  from sklearn.model_selection import train_test_split

    new_df = pd.DataFrame(processed_data)

    train_df, valid_df = train_test_split(
        new_df,
        test_size=0.2,
        random_state=2022
    )

    Convert the train_df and valid_df into Dataset accepted by the Trainer module.

    import pyarrow as pa
    from datasets import Dataset

    train_hg = Dataset(pa.Table.from_pandas(train_df))
    valid_hg = Dataset(pa.Table.from_pandas(valid_df))

    Train and Evaluate the model

    Since the dataset is now ready, we can safely move forward to training the model on our custom dataset.

The following piece of code will initialize a Bert base uncased model for our training.

 from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    The model is ready, now create a Trainer, it will require TrainingArguments as well. The following code will initialize both of them.
    Under TrainingArguments you will see a outpit_dir argument, it is used by the module to write training logs.

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(output_dir="./result", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hg,
        eval_dataset=valid_hg,
        tokenizer=tokenizer
    )

    Training the model

    trainer.train()

    Once the training is complete, we can evaluate the model as well.

    trainer.evaluate()

    Test the model

    Save the model
    save the model at the desired location.
    
    model.save_pretrained('./model/')

    Load the model

    To load the model and initialize the tokenizer

    from transformers import AutoModelForSequenceClassification

    new_model = AutoModelForSequenceClassification.from_pretrained('./model/')


    from transformers import AutoTokenizer

    new_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


    Get predictions

    The following function will use the model and tokenizer to get the prediction from a piece of text.
    import torch
    import numpy as np

    def get_prediction(text):
        encoding = new_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

        outputs = new_model(**encoding)

        logits = outputs.logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        probs = probs.detach().numpy()
        label = np.argmax(probs, axis=-1)
        
        if label == 1:
            return {
                'sentiment': 'Positive',
                'probability': probs[1]
            }
        else:
            return {
                'sentiment': 'Negative',
                'probability': probs[0]
            }


    get_prediction('I am happy to see you.')



Challenges and Limitations

Hyperparameter tuning: Finding the optimal hyperparameters required experimentation and careful evaluation.
Limited training data: Larger datasets could potentially improve model performance.
Overfitting: The model showed signs of overfitting, requiring regularization techniques like dropout.
