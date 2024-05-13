# Human_Activity_Video_Recognition

- [ ] Add hugging face models
- [x] Add proper model logging
- [ ] Train model properly
- [x] Add elas names to confusion matrix
- [x] Add training history plotting
- [x] Add training history comparison dataset
- [x] No EARLY STOP HMDB
- [x] No EARLY STOP UFC
- [x] No EARLY STOP CUSTOM DATASET
- [ ] Look into generator for input data, to lower memory issues
- [ ] Look into GPU training setup
- [x] Implement single class prediction
- [x] Implement multi class multi class confidence percentage prediction
- [ ] Look into model architecture changes
- [ ] Look into regularizers L1, L2, Dropout and early stopping,
- [ ] Implement camera feed recognition
- [x] Add grid search training
- [ ] Implement CLI

transfer learning

# Usage

## Help

   ```sh
   python main.py --help
   ```

## Training

   ```sh
   python main.py --functionality train --model convlstm --dataset dvorak_custom --epochs 50 --test-percentage 0.2 --resolution 224 --batch-size 32
   ```

   ```sh
   python main.py --functionality train --model resnet --dataset dvorak_custom --epochs 50 --test-percentage 0.2 --resolution 224 --batch-size 32
   ```

   ```sh
   python main.py --functionality train --model googlenet --dataset dvorak_custom --epochs 50 --test-percentage 0.2 --resolution 224 --batch-size 32
   ```

   ```sh
   python main.py --functionality train --model timesformer --dataset dvorak_custom --epochs 50 --test-percentage 0.2 --resolution 224 --batch-size 32
   ```

## Prediction
   ```sh
   python main.py --functionality predict --single-class-prediction --model MODEL_TYPE --model-path "MODEL_PATH.keras" --paths "['VIDEO_PATH.mp4']"
   ```
