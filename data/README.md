# Dataset

UCI Human Activity Recognition Using Smartphones (v1.0).

30 subjects, 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING), 561 features from accelerometer and gyroscope signals.

Source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

I use the original subject-aware split (21 train / 9 test). Validation is stratified 5-fold within training subjects. All preprocessing is fit on training data only.

Run `uv run python src/data.py --download` to fetch the data. Raw and processed files are gitignored.
