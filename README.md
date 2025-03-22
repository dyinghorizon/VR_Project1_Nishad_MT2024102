# VR_Project1_Nishad_MT2024102

Based on the PDF document you shared, here's the "How to Run the Code" section in markdown format:

```markdown
# How to Run the Code

The face mask detection system is implemented in Python with the following dependencies:

- NumPy, Pandas
- scikit-image, OpenCV
- scikit-learn
- Matplotlib, Seaborn

The implementation follows these steps:

1. Extract features from images using HOG, LBP, color histograms, and Haralick features
2. Apply PCA for dimensionality reduction
3. Train and evaluate SVM, Neural Network, and Random Forest classifiers
4. Compare model performance metrics

## For CNN-based approach:

To run the CNN-based face mask detection system, follow these steps:

1. Ensure you have PyTorch, torchvision, and other required libraries installed:
   ```
   pip install torch torchvision numpy pandas matplotlib scikit-learn pillow
   ```

2. Organize your dataset into the following structure:
   ```
   dataset/
       with_mask/
           img1.jpg
           img2.jpg
           ...
       without_mask/
           img1.jpg
           img2.jpg
           ...
   ```

3. Run the CNN training script:
   ```
   python cnn_mask_detection.py --dataset_path path/to/dataset
   ```

4. To experiment with different hyperparameters:
   ```
   python cnn_mask_detection.py --dataset_path path/to/dataset --batch_size 16 --learning_rate 0.001 --dropout 0.3 --epochs 10 --optimizer adam
   ```

5. To run the binary classification model with BCEWithLogitsLoss:
   ```
   python cnn_mask_detection.py --dataset_path path/to/dataset --model binary --batch_size 16 --optimizer sgd
   ```
```
