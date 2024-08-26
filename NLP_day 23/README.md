# Sentiment Analysis of Dune Part 2 Reviews on IMDb Using LSTM

## Project Description

This project aims to analyze the sentiment of reviews for the film *Dune Part 2* on IMDb using a Long Short-Term Memory (LSTM) model. The analysis involves several stages, from data collection to model evaluation.

## 1. Scraping

1. **Data Collection**:
   - Used a saved HTML file to scrape reviews for *Dune Part 2* from IMDb.
   - Utilized BeautifulSoup to parse the HTML content and extract elements with the class `review-container`.
   - Collected review text and stored it in a list called `reviews`.

## 2. Creating a DataFrame

1. **Data Storage**:
   - Created a DataFrame from the collected reviews.
   - Saved the DataFrame to a CSV file named `IMDB_Reviews_Dune.csv`.
   - Loaded the CSV file into a variable `data` using `pd.read_csv()`.

## 3. Preprocessing

1. **Data Cleaning**:
   - Removed newline characters (`\n`) from each line in the `Review` column.
   - Applied case folding to convert all text to lowercase.
   - Removed numbers and punctuation.
   - Removed stop words from the text.
   - Created a tokenizer object using `BertTokenizer.from_pretrained('bert-base-uncased')`.

## 4. Sentiment Analysis

1. **Sentiment Analysis**:
   - Used TextBlob to analyze the sentiment of the text.
   - Calculated sentiment polarity, which ranges from -1 (very negative) to 1 (very positive).

## 5. Creating Labels X and y

1. **Token Conversion and Padding**:
   - Converted tokens to IDs using `tokenizer.convert_tokens_to_ids()`.
   - Applied padding to token sequences using `pad_sequences()`.
   - Extracted sentiment labels and converted them to numeric values ('Positive' = 0, 'Negative' = 1, 'Neutral' = 2).

## 6. LSTM Model

1. **Model Building**:
   - Used the `Sequential` model from Keras.
   - Added an `Embedding` layer to convert token-IDs into embedding vectors.
   - Added an `LSTM` layer to capture long-term and short-term dependencies.
   - Added a `Dense` layer with a `softmax` activation function to output probabilities for each sentiment class.

## 7. Sanity Check

1. **Preprocessing and Prediction Functions**:
   - `preprocess_text`: Function for preprocessing text before making predictions.
   - `predict`: Function for predicting sentiment based on the given text.
