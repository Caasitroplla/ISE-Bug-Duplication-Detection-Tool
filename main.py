import pandas as pd
from transformers import BertTokenizer
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer


def read_csv_file(csv_file_path = 'eclipse_platform.csv'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, encoding='utf-8')

    # First row is the header
    #df.columns = df.iloc[0]
    #df = df.iloc[1:]

    df.columns = df.columns.str.strip()

    return df


# Class representing a bug report
class BugReport:
    def __init__(self, issue_id, priority, component, duplicated_issue, title, description, status, resolution, version, created_time, resolved_time):
        self.issue_id = int(issue_id)
        self.priority = str(priority)
        self.component = str(component)
        # If duplicated issue is Nan, set it to 0
        self.duplicated_issue = int(duplicated_issue) if pd.notna(duplicated_issue) else 0
        self.title = str(title)
        self.description = str(description)
        self.resolution = str(resolution)
        self.version = str(version)
        self.created_time = str(created_time)
        self.resolved_time = str(resolved_time)
        self.processed = []
        self.tokenised = []

    def preprocess(self):
        # For each text variable we need to combine them into a single string, then preprocess it then tokenise
        self.processed = self.processed + self.title.split()
        self.processed = self.processed + self.description.split()
        self.processed = self.processed + self.resolution.split()
        self.processed = self.processed + self.version.split()

        # Create a new list to store processed words
        processed_words = []

        # For each word in the processed list:
        for word in self.processed:
            word = word.lower()
            # Remove punctuation
            word = ''.join(char for char in word if char not in punctuation)
            # If word is not a stop word and not a number, add it to processed_words
            if word and word not in stopwords.words('english') and not word.isdigit():
                processed_words.append(word)

        # Update self.processed with non-empty words
        self.processed = processed_words

        # Tokenise each word 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for word in self.processed:
            tokenised_word = tokenizer.tokenize(word)
            for token in tokenised_word:
                self.tokenised.append(token)


    def __str__(self):
        return f"Bug Report ID: {self.issue_id}\nTitle: {self.title}\nDescription: {self.description}\nResolution: {self.resolution}\nVersion: {self.version}\nCreated Time: {self.created_time}\nResolved Time: {self.resolved_time}\nProcessed: {self.processed}\nTokenised: {self.tokenised}"

    def is_duplicate(self, other_bug_issue_id) -> bool:
        return self.duplicated_issue == other_bug_issue_id


class BugReportDuplicate:
    def __init__(self, df):
        self.df = df
        self.train_df = None
        self.test_df = None

    # Not necessary
    def put_into_BugReport(self):
        # Process all rows with progress bar
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Bug Reports"):
            bug_report = BugReport(row['Issue_id'], row['Priority'], row['Component'], row['Duplicated_issue'], row['Title'], row['Description'], row['Status'], row['Resolution'], row['Version'], row['Created_time'], row['Resolved_time'])
            bug_report.preprocess()

        # Test for first 3 rows
        # for i in range(3):
        #     bug_report = BugReport(self.df.iloc[i]['Issue_id'], self.df.iloc[i]['Priority'], self.df.iloc[i]['Component'], self.df.iloc[i]['Duplicated_issue'], self.df.iloc[i]['Title'], self.df.iloc[i]['Description'], self.df.iloc[i]['Status'], self.df.iloc[i]['Resolution'], self.df.iloc[i]['Version'], self.df.iloc[i]['Created_time'], self.df.iloc[i]['Resolved_time'])
        #     bug_report.preprocess()
        #     print(bug_report)

    def split_into_train_test(self):
        # Split the dataframe into train and test
        self.train_df = self.df.sample(frac=0.8, random_state=42)
        self.test_df = self.df.drop(self.train_df.index)
        self.model = None
  

    def train_model(self):
        # Prepare data
        pairs, labels = self.create_pairs_and_labels(self.train_df)

        # Tokenize and pad sequences
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(pairs.flatten())
        sequences = tokenizer.texts_to_sequences(pairs.flatten())
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        # Reshape data for CNN input
        X = padded_sequences.reshape(-1, 2, max_length)
        y = np.array(labels)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build CNN model
        self.model = Sequential([
            Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        print("Training model...")
        self.model.fit([X_train[:, 0], X_train[:, 1]], y_train, validation_data=([X_val[:, 0], X_val[:, 1]], y_val), epochs=10, batch_size=32)
        print("Model trained")

    def create_pairs_and_labels(self, df):
        pairs = []
        labels = []
        
        # Add tqdm progress bar to the outer loop
        for i in tqdm(range(len(df)), desc="Creating Pairs and Labels"):
            for j in range(i + 1, len(df)):
                bug1 = BugReport(df.iloc[i]['Issue_id'], df.iloc[i]['Priority'], df.iloc[i]['Component'], df.iloc[i]['Duplicated_issue'], df.iloc[i]['Title'], df.iloc[i]['Description'], df.iloc[i]['Status'], df.iloc[i]['Resolution'], df.iloc[i]['Version'], df.iloc[i]['Created_time'], df.iloc[i]['Resolved_time'])
                bug2 = BugReport(df.iloc[j]['Issue_id'], df.iloc[j]['Priority'], df.iloc[j]['Component'], df.iloc[j]['Duplicated_issue'], df.iloc[j]['Title'], df.iloc[j]['Description'], df.iloc[j]['Status'], df.iloc[j]['Resolution'], df.iloc[j]['Version'], df.iloc[j]['Created_time'], df.iloc[j]['Resolved_time'])
                
                # Preprocess both bug reports
                bug1.preprocess()
                bug2.preprocess()

                # Create a pair of processed text
                pairs.append((bug1.processed, bug2.processed))
                
                # Use is_duplicate to determine the label
                labels.append(1 if bug1.is_duplicate(bug2.issue_id) or bug2.is_duplicate(bug1.issue_id) else 0)

        return np.array(pairs), labels
    
    def test_model(self):
        # Prepare data
        pairs, labels = self.create_pairs_and_labels(self.test_df)

        # Tokenize and pad sequences
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(pairs.flatten())
        sequences = tokenizer.texts_to_sequences(pairs.flatten())
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        # Reshape data for CNN input
        X_test = padded_sequences.reshape(-1, 2, max_length)
        y_test = np.array(labels)

        # Evaluate the model
        print("Testing model...")
        loss, accuracy = self.model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, batch_size=32)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def save_model(self):
        # Save the model
        if self.model is not None:
            self.model.save('bug_report_duplicate_model.keras')
            print("Model saved")
        else:
            print("Model is not trained yet")


if __name__ == '__main__':
    df = read_csv_file()
    bug_report_duplicate = BugReportDuplicate(df)
    bug_report_duplicate.split_into_train_test()
    bug_report_duplicate.train_model()
    bug_report_duplicate.test_model()
    bug_report_duplicate.save_model()
 