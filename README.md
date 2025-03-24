# Project Overview
This project involves watershed-based segmentation, binary classification, and U-Net-based mask segmentation.

## Project Authors
- Nishad Bagade (MT2024102)
- Sriram Bharadwaj (MT2024114)

## Directory Structure
```
.
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
├── VR_Project_Binary_Classification.ipynb  # Binary classification notebook
├── best_watershed_params.txt      # Best parameters for watershed segmentation
├── best_wateshed.py               # Python script for optimized watershed segmentation
├── requirements.txt               # List of dependencies
├── unet-mask-segmentation.ipynb   # Notebook for U-Net based segmentation
├── unet.py                        # U-Net model implementation
├── watershed.py                   # Watershed segmentation script
```

## Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### 1. Binary Classification
Run the Jupyter Notebook:
```sh
jupyter notebook VR_Project_Binary_Classification.ipynb
```

### 2. Watershed Segmentation
Execute the watershed segmentation script:
```sh
python watershed.py
```
For optimized parameters:
```sh
python best_wateshed.py
```

### 3. U-Net Mask Segmentation
Run the U-Net segmentation notebook:
```sh
jupyter notebook unet-mask-segmentation.ipynb
```
Or execute the U-Net model:
```sh
python unet.py
```

## Notes
- Ensure all dependencies are installed before running the scripts.
- Modify parameters in `best_watershed_params.txt` if needed.

## License
This project is under the MIT License.

## Acknowledgments
- OpenCV for image processing
- TensorFlow/PyTorch for deep learning models

