import os

import pandas as pd

dataset_path = os.listdir('dataset/train')

print(dataset_path)
# Getting the directory labels inside train folder

# Preparing training data
rooms = []

for item in dataset_path:
    # Get all the file names
    all_rooms = os.listdir('dataset/train' + '/' + item)

    # Add them to the list
    for room in all_rooms:
        rooms.append((item, str('dataset/train' + '/' + item) + '/' + room))

# Build a dataframe
train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
print(train_df.head())
print(train_df.tail())


df = train_df.loc[:, ['video_name', 'tag']]
df
df.to_csv('train.csv')

# Preparing test data

dataset_path = os.listdir('dataset/test')
print(dataset_path)

room_types = os.listdir('dataset/test')
print('Types of activities found: ', len(dataset_path))

rooms = []

for item in dataset_path:
    # Get all the file names
    all_rooms = os.listdir('dataset/test' + '/' + item)

    # Add them to the list
    for room in all_rooms:
        rooms.append((item, str('dataset/test' + '/' + item) + '/' + room))

# Build a dataframe
test_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
print(test_df.head())
print(test_df.tail())

df = test_df.loc[:, ['video_name', 'tag']]
df
df.to_csv('test.csv')
