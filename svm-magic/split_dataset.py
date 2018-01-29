import os
import json

train_paths = json.load(open('./food101/meta/train.json'))
test_paths = json.load(open('./food101/meta/test.json'))
foldernames = open('./food101/meta/classes.txt')

# # Make folders for every class
# for folder in foldernames:
#     os.mkdir('./food101/train/'+folder.strip()+'/')
#     os.mkdir('./food101/test/'+folder.strip()+'/')

for folder in os.listdir('./food101/images/'):
    for filepath in os.listdir('./food101/images/'+folder):

        compatible_filepath = folder+'/'+filepath.replace('.jpg', '')
        complete_filepath = './food101/images/'+folder+'/'+filepath
        train_filepath = './food101/train/'+folder+'/'+filepath
        test_filepath = './food101/test/'+folder+'/'+filepath

        # Check if it is in test
        if compatible_filepath in test_paths[folder]:

            # Move it to the train thing
            os.rename(complete_filepath, test_filepath)

        else:
            os.rename(complete_filepath, train_filepath)


