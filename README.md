# VR_Project1_Nishad_MT2024102

# How to Run the Code

The face mask detection system is implemented as a Jupyter notebook with the following dependencies:

- NumPy, Pandas
- scikit-image, OpenCV
- scikit-learn
- Matplotlib, Seaborn
- PyTorch (for CNN implementation)

## Running the code:

1. Ensure all dependencies are installed in your Python environment
2. Open the notebook file (VR_Project_Part(i).ipynb) in Jupyter Notebook or JupyterLab
3. Run the code cells sequentially from top to bottom
4. The implementation flow follows these steps:
   - Import libraries
   - Load and visualize dataset
   - Extract features (HOG, LBP, color histograms, Haralick)
   - Apply PCA for dimensionality reduction
   - Train and evaluate SVM, Neural Network, and Random Forest classifiers
   - Compare model performance metrics

For the CNN implementation:
- Additional cells implement different CNN architectures
- Run these cells sequentially after the initial feature-based approach
- The notebook includes various experimental configurations for hyperparameter tuning
