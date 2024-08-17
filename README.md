# Bachelor Thesis Project: EEG Data Analysis with CHB-MIT Dataset

This repository contains the code, data, and reports related to my bachelor thesis project, which focuses on EEG data analysis using the CHB-MIT dataset. The project is organized into several stages, with the current focus on preprocessing and feature extraction.

## Repository Structure

The repository is organized into the following folders:

- **Codes**: Contains all the Jupyter Notebooks used in this project, including scripts for preprocessing, feature extraction, and more.
- **Sample Data**: Includes sample EEG data from the CHB-MIT dataset used for testing and development purposes.
- **Preprocessed Data**: Stores the EEG data after it has been preprocessed, ready for feature extraction or further analysis.
- **Presentations and Reports**: Contains weekly reports, presentations, and documentation that outline the progress and findings of the project.

## Data Source

The project utilizes the **CHB-MIT EEG dataset**, a publicly available dataset that consists of EEG recordings from pediatric patients with intractable seizures. The dataset is widely used for epilepsy research and seizure detection studies.

## Current Progress

### 1. Preprocessing

The preprocessing stage involves cleaning and preparing the raw EEG data for analysis. The steps include:

- **Artifact Removal**: Identifying and removing artifacts such as eye blinks, muscle movements, and other noise from the EEG signals.
- **Filtering**: Applying band-pass filters to isolate the relevant frequency bands for seizure detection (e.g., delta, theta, alpha, beta).
- **Normalization**: Standardizing the EEG signals to ensure consistency across different recordings and subjects.

**Notebook**: `Preprocessing.ipynb`

### 2. Feature Extraction

Feature extraction is the process of transforming the preprocessed EEG data into a set of meaningful features that can be used for further analysis, such as machine learning models for seizure detection. The steps include:

- **Time-Domain Features**: Extracting features such as mean, variance, skewness, and kurtosis from the EEG signals.
- **Frequency-Domain Features**: Applying techniques like Fourier Transform to extract features related to the frequency components of the EEG signals.
- **Entropy Measures**: Calculating the entropy of the EEG signals to quantify the complexity and irregularity of the brain activity.

**Notebook**: `Feature_Extraction.ipynb`

## Usage

### Prerequisites

Before running the notebooks, ensure that you have the necessary Python libraries installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. **Preprocessing**:
   - Navigate to the `Codes` folder and open `Preprocessing.ipynb`.
   - Follow the instructions in the notebook to preprocess the raw EEG data.
   - The preprocessed data will be saved in the `Preprocessed Data` folder.

2. **Feature Extraction**:
   - Open the `Feature_Extraction.ipynb` notebook in the `Codes` folder.
   - Execute the cells to extract features from the preprocessed EEG data.
   - The extracted features can be saved for later use in machine learning models or further analysis.

## Sample Data

The `Sample Data` folder contains a subset of the CHB-MIT EEG dataset used for testing and development. This data can be used to understand and verify the preprocessing and feature extraction steps before applying them to the full dataset.

## Reports and Presentations

The `Presentations and Reports` folder includes all the documentation related to this project. This includes:

- **Weekly Reports**: Detailed reports outlining the progress made each week, challenges encountered, and solutions implemented.
- **Presentations**: Slides from presentations given during the project, summarizing key findings and methodologies.

## Future Work

- **Model Development**: In upcoming stages, I plan to develop machine learning models for seizure detection using the extracted features.
- **Validation and Testing**: The models will be validated and tested on the full CHB-MIT dataset to evaluate their performance.
- **Further Feature Engineering**: Additional features and methods will be explored to improve the accuracy and robustness of the seizure detection models.

## Acknowledgments

I would like to thank my advisors, Dr. Farnaz Ghassemi, Dr. Fatemeh Zare and Dr. Zahra Tabanfar, and the both faculties of Biomedical Engineering and Computer Science at Amirkabor University of Technology for their guidance and support throughout this project.
