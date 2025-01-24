import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization

#Mapping numeric codes to emotion labels based on RAVDESS naming conventions
emotion_mapping = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def extract_features(audio_segment, sample_rate=16000):
    try:
        #Extracting MFCCs and their deltas
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        #Extracting Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sample_rate)

        #Extracting Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sample_rate)

        #Extracting RMS Energy
        rms = librosa.feature.rms(y=audio_segment)

        #Extracting Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_segment)

        #Combining all features
        combined_features = np.concatenate((
            np.mean(mfcc.T, axis=0), np.mean(mfcc_delta.T, axis=0), np.mean(mfcc_delta2.T, axis=0),
            np.mean(chroma.T, axis=0), np.mean(spectral_contrast.T, axis=0),
            [np.mean(rms)], [np.mean(zcr)]
        ))
        return combined_features
    except Exception as e:
        print(f"Error extracting features from audio segment: {e}")
        return None



#Function to load dataset and extract features
def load_dataset(dataset_path):
    data = []
    labels = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    emotion_code = file.split("-")[2]
                    emotion_label = emotion_mapping.get(emotion_code, None)
                    if emotion_label:
                        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
                        features = extract_features(audio, sample_rate)
                        if features is not None:
                            data.append(features)
                            labels.append(emotion_label)
                    else:
                        print(f"Skipping file (unknown emotion): {file_path}")
                except Exception as e:
                    print(f"Error processing file: {file_path}, Error: {e}")
    return np.array(data), np.array(labels)


#Function to process audio in overlapping windows
def process_audio_in_segments(audio_path, window_size=2.0, overlap=1.0):
    try:
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        step = window_size - overlap
        segments = []
        segment_times = []
        for start in np.arange(0, duration, step):
            end = min(start + window_size, duration)
            audio_segment = audio[int(start * sample_rate):int(end * sample_rate)]
            if len(audio_segment) > 0:
                features = extract_features(audio_segment, sample_rate)
                if features is not None:
                    segments.append(features)
                    segment_times.append((start, end))
        return np.array(segments), segment_times
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return np.array([]), []


# Building an RNN model with Bidirectional LSTM
def build_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    dataset_path = "archive"
    print("Loading dataset..")
    data, labels = load_dataset(dataset_path)

    if len(data) == 0 or len(labels) == 0:
        raise ValueError("No data or labels found. check dataset path and file compatibility.")

    #Standardizing features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    #Encoding labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    #Reshaping data for RNN input
    data = np.expand_dims(data, axis=1)

    #Splitting dataset into training and testing sets
    print("Splitting dataset into training and testing sets..")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    #Building and training the RNN model
    print("Building and training the RNN model..")
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    model = build_rnn_model(input_shape, num_classes)

    #Training with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

    #Evaluating the RNN model
    print("Evaluating the RNN model..")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    #Detecting emotions dynamically
    test_audio_path = "audio.mp3"
    print("Detecting emotions dynamically..")
    segments, segment_times = process_audio_in_segments(test_audio_path)
    if len(segments) > 0:
        segments = np.expand_dims(segments, axis=1)
        predicted_emotions = model.predict(segments)
        predicted_emotions = np.argmax(predicted_emotions, axis=1)
        for (start, end), emotion_idx in zip(segment_times, predicted_emotions):
            emotion = label_encoder.inverse_transform([emotion_idx])[0]
            print(f"From {start:.2f}s to {end:.2f}s: {emotion}")
