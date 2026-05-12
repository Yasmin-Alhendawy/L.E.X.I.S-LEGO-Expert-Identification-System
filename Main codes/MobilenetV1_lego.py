import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from time import perf_counter
from sklearn.metrics import confusion_matrix, accuracy_score
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

# Load filepaths and labels
dir_ = Path(r"C:\Users\Jasmi\Desktop\Machine project\Meow\Data5")
file_paths = list(dir_.glob(r'**/*.png'))
df = pd.DataFrame({'Filepath': [str(x) for x in file_paths]})

def get_label(string):
    return ' '.join(string.split('/')[-1].replace('.png', '').split(' ')[1:-1]).lower()

df['Label'] = df['Filepath'].apply(get_label)

# Visualization
fig, axes = plt.subplots(4, 6, figsize=(15, 7), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df.Filepath[i]))
    ax.set_title(df.Label[i], fontsize=15)
plt.tight_layout(pad=0.5)
plt.show()

vc = df['Label'].value_counts()
plt.figure(figsize=(20,5))
sns.barplot(x=sorted(vc.index), y=vc, palette="rocket")
plt.title("Number of pictures of each category", fontsize=15)
plt.xticks(rotation=90)
plt.show()

# Load validation set
validation = pd.read_csv('../input/lego-brick-images/validation.txt', names=['Filepath'])
validation['Filepath'] = validation['Filepath'].apply(lambda x: '../input/lego-brick-images/dataset/' + x)
df['validation_set'] = df['Filepath'].isin(validation['Filepath'])

# Split into training and testing
train_df = df[df['validation_set'] == False].sample(frac=0.3, random_state=42)
test_df = df[df['validation_set'] == True].sample(frac=0.3, random_state=42)
printmd(f'### Number of pictures in the train set: {train_df.shape[0]}')
printmd(f'### Number of pictures in the test set: {test_df.shape[0]}')

# Data Generators
def create_gen():
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
        validation_split=0.1)
    
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

    train_images = train_gen.flow_from_dataframe(
        dataframe=train_df, x_col='Filepath', y_col='Label',
        target_size=(224, 224), class_mode='categorical', batch_size=32,
        subset='training', shuffle=True, seed=42,
        rotation_range=30, zoom_range=0.15, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode='nearest')

    val_images = train_gen.flow_from_dataframe(
        dataframe=train_df, x_col='Filepath', y_col='Label',
        target_size=(224, 224), class_mode='categorical', batch_size=32,
        subset='validation', shuffle=True, seed=42,
        rotation_range=30, zoom_range=0.15, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode='nearest')

    test_images = test_gen.flow_from_dataframe(
        dataframe=test_df, x_col='Filepath', y_col='Label',
        target_size=(224, 224), class_mode='categorical', batch_size=32, shuffle=False)

    return train_images, val_images, test_images

train_images, val_images, test_images = create_gen()

# Model with MobileNet
def build_mobilenet_model(num_classes):
    base = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')
    base.trainable = False
    x = tf.keras.layers.Dense(128, activation='relu')(base.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_mobilenet_model(train_images.num_classes)
history = model.fit(
    train_images, validation_data=val_images, epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Plot metrics
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(ax=axes[0])
axes[0].set_title("Accuracy", fontsize=15)
axes[0].set_ylim(0, 1.1)
pd.DataFrame(history.history)[['loss','val_loss']].plot(ax=axes[1])
axes[1].set_title("Loss", fontsize=15)
plt.tight_layout()
plt.show()

# Evaluation
pred = np.argmax(model.predict(test_images), axis=1)
label_map = {v: k for k, v in train_images.class_indices.items()}
pred_labels = [label_map[k] for k in pred]
y_true = list(test_df.Label)
acc = accuracy_score(y_true, pred_labels)
printmd(f'## MobileNet Accuracy on test set: {acc * 100:.2f}%')

cf_matrix = confusion_matrix(y_true, pred_labels, normalize='true')
plt.figure(figsize=(20,15))
sns.heatmap(cf_matrix, annot=True, xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)), cbar=False)
plt.title('Normalized Confusion Matrix', fontsize=23)
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.show()

# Show predictions
fig, axes = plt.subplots(4, 6, figsize=(20, 12), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
    ax.set_title(f"True: {y_true[i].split(' ')[0]}\nPredicted: {pred_labels[i].split(' ')[0]}", fontsize=15)
plt.tight_layout()
plt.show()
