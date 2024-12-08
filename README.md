# Magellan
**Automated Flood Detection Using Satellite Images**  
**Budapest University of Technology and Economics (BME)**

## Team Name:
**Magellan**

## Team Members:
- **Tóth Ádám László** (ID: TK6NT3)
- **Szladek Máté Nándor** (ID: TGPZTT)

 This project focuses on the processing of Sentinel-2 satellite imagery from the SEN12-FLOOD database and flood detection. Four spectral bands are used for processing: Band2 (blue), Band3 (green), Band4 (red) and Band8 (near-infrared). These bands are of particular importance for distinguishing land, water and other surface features. Throughout the project, special attention was paid to data organisation, filtering out blank images and band alignment to provide a suitable basis for flood detection tasks.
<hr>

## Data Processing and Model Development

### 1. Data Quality and Challenges
The quality of the processed data significantly impacts model performance. The SEN12-FLOOD database, while comprehensive, does not always provide sufficient resolution and quality for flood detection. Contours of flooded areas are often blurred, and noise hampers analysis, making it challenging for advanced models to cluster classes accurately.

### 2. Developing the Base Model
The base model serves as the starting point for the project, illustrating fundamental principles of flood detection and providing a reference for further development. With its simpler architecture, it helps to:
- Understand the data.
- Measure augmentation impacts.
- Determine directions for parameter tuning.

### 3. Advanced Model and Its Limitations
The advanced model incorporates modern techniques to optimize flood detection. However, its classification results fall short due to data quality and resolution issues, emphasizing the importance of data quality control.

### Key Evaluation Steps and Metrics

| **Evaluation Steps**             | **Details**                                                                                                                                                                                                                                      |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Spectral Band Statistics**     | - Mean and Normalized Standard Deviation (Mean_Std).<br> - Standard Deviation (Std_Dev) and Normalized Standard Deviation (Std_Dev_Std).<br> - Mean Contribution (Mean_Contribution).<br> - Standard Deviation-Based Contribution (Std_Contribution). |
| **Class Distribution Analysis**  | - Distribution of data classes: CLEAR and FLOODED.<br> - Class balance assessment with sample proportions.                                                                                                                                       |
| **Visualizations**               | - Pie Chart: Spectral band contributions (Blue, Green, Red, NIR).<br> - Confusion Matrix: Classification performance.<br> - ROC Curve: Classification evaluation.<br> - PR Curve: Precision and sensitivity comparison.                          |
| **Performance Metrics**          | - Accuracy<br> - Precision<br> - Recall<br> - F1-score<br> - Class-specific summaries                                                                                                                                                            |
| **Relative Contribution Analysis** | - Quantification of average and standard deviation contributions for each spectral band.<br> - Identification of dominant bands (e.g., NIR).                                                                                                     |

### Additional Features
- **Wandb Integration**: Logging is included but requires your own API key.
- **Gradio Interface**: Allows testing the model on uploaded files with prediction results.


<hr>

### How to run
The repository contains four Docker Hub links:

The first one describes data processing. The first one shows the first one, which contains the original data. The first one is the original data. Due to the size of the data, the other DockerHub images contain only the processed, carefully selected data.

Data is not accessible without authentication, and no in-code authentication is enabled. Data can be accessed after free registration (without any permission), but due to the free accessibility, data is inside the image file.

The file operations are platform independent, so the image file can be downloaded and run on a windows system without any permission. 

DockerHub public links:

- [Magellan_RAW](https://hub.docker.com/repository/docker/tgpztt/magellan_raw/general)
- [Magellan](https://hub.docker.com/repository/docker/tgpztt/magellan/general)
- [Magellan_Milestone2](https://hub.docker.com/repository/docker/tgpztt/magellan_milestone2) (contains baseline model, augmentation, JSON filtering)



# Project Instructions - How to Run the Project

| **Step** | **Title**              | **Instructions** |
|----------|-------------------------|-------------------|
| **A**    | **Docker Image (Cross-Platform)** | **To Run the Project**:<br><br>1. **Pull the chosen Docker Image:**<br>Use the following command to pull the desired Docker image from Docker Hub:<br><br>`docker pull tgpztt/magellan_milestone2:latest`<br><br>2. **Run the Docker Container:**<br>Execute the following command to run the Docker image, ensuring to map port 8888 for Jupyter Notebook access:<br><br>`docker run -p 8888:8888 tgpztt/magellan_milestone2:latest`<br><br>3. **Access Jupyter Notebook:**<br>Once the co...
| **B**    | **Data Preparation**    | 1. Download the dataset from the IEEE website (link is in the "Related Work and Papers" section).<br><br>2. Download `final.ipynb` file from this repository.<br><br>3. Ensure that the downloaded dataset and the `.ipynb` file are in the same folder.<br><br>4. Open `final.ipynb` and run the first blocks. Instructions are provided in the markdown cells within the notebook. |
| **C**    | **For Training**        | 1. Download the prepared dataset from this repository.<br><br>2. Download `final.ipynb` file from this repository.<br><br>3. Ensure that the `.ipynb` file and the `SEN12FLOOD` folder (containing 2037 subfolders and JSON files) are in the same folder.<br><br>4. Open `final.ipynb` and run the block after the **"Model"** title. Instructions are provided in the markdown cells within the notebook. |
| **D**    | **For Evaluating**      | 1. Download the model file (`flood_model.pth`) from this repository.<br><br>2. Download `final.ipynb` file from this repository.<br><br>3. Ensure that the `.ipynb` file and the model file are in the same folder.<br><br>4. Open `final.ipynb` and run the block after the **"Eval"** title. Instructions are provided in the markdown cells within the notebook. |

---

**Note:**<br>
B, C, and D steps are using the same `magellan.ipynb` file. You can run all the tasks above if you have generated or downloaded the files needed.<br>
If you are using B, C, or D methods, be sure you have installed the required packages!

<hr>

# Running environment and components


|              | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                 |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Programming Language**   | Python 3.11.5                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Jupyter Notebook**       | Jupyter Notebook is used for interactive data processing and analysis. If you have pulled the docker file it runs on port 8888.                                                                                                                                                                                                                                                                                               |
| **Dependencies**           | **_File and Directory Management_**<br> - `os` - File and directory operations.<br> - `shutil` - File manipulation (e.g., copy, move, delete).<br> - `json` - Reading and writing JSON files.<br><br> **_Data Manipulation_**<br> - `numpy` - Numerical computations and array manipulation.<br> - `pandas` - Dataframe and tabular data handling.<br> - `random` - Random number generation.<br><br> **_Image Handling and Processing_**<br> - `cv2` - Image processing and computer vision.<br> - `Pillow [Image]` - Opening and manipulating image files.<br> - `tifffile` - Handling TIFF image files.<br><br> **_Visualization_**<br> - `matplotlib.pyplot` - General-purpose plotting and visualization.<br> - `seaborn` - Statistical data visualization.<br><br> **_Deep Learning (PyTorch Ecosystem)_**<br> - `torch` - Core library for tensors and models.<br> - `torch.nn.functional [F]` - Functional utilities for neural networks.<br> - `torchvision` - Pre-trained models, datasets, and transforms.<br> - `torch.utils.data [WeightedRandomSampler]` - Weighted sampling for datasets.<br> - `torchmetrics` - Metrics for model evaluation.<br> - `pytorch_lightning` - High-level PyTorch training abstraction.<br> - `pytorch_lightning.callbacks [EarlyStopping]` - Early stopping callback.<br> - `pytorch_lightning [Trainer]` - Manages PyTorch training loops.<br><br> **_Deep Learning (TensorFlow and Keras Ecosystem)_**<br> - `tensorflow` - Core library for deep learning.<br> - `tensorflow.keras.models [Sequential]` - Model building.<br> - `tensorflow.keras.layers` - Includes `[Dense]`, `[Conv2D]`, `[MaxPool2D]`, `[Flatten]`, and `[Dropout]` for neural network layers.<br> - `tensorflow.keras.preprocessing.image [ImageDataGenerator]` - Data augmentation.<br> - `tensorflow.keras.optimizers [Adam]` - Adam optimizer.<br> - `tensorflow.keras.utils [plot_model]` - Visualize model architecture.<br><br> **_Machine Learning Metrics_**<br> - `sklearn.metrics` - Includes `[classification_report]`, `[confusion_matrix]`, `[roc_curve]`, `[auc]`, `[precision_recall_curve]`, and `[average_precision_score]` for evaluating model performance.<br><br> **_Experiment Tracking and Interface_**<br> - `wandb` - Experiment tracking and visualization.<br> - `gradio` - Build user interfaces for machine learning models. |
| **Application Purpose**    | This project involves geospatial data processing and flood detection, relying on raster data manipulation, image augmentation, and a baseline model for flood classification.                                                                                                                                                                                                                                                   |
| **Project Access**         | If you have pulled a Docker image, the Jupyter Notebook environment is accessed through a web browser at [http://localhost:8888](http://localhost:8888).                                                                                                                                                                                                                                                                       |
| **Warning**                | **Do not install any additional software or dependencies** within the environment, as everything needed is already pre-installed in the Docker image.                                                                                                                                                                                                                                                                         |


<hr>

**Related Works and Papers:**

|**Type**|**Title**|**Authors**|**Year**|**Link**|
| :- | :- | :- | :- | :- |
|**Dataset**|SEN12-FLOOD: a SAR and Multispectral Dataset for Flood Detection|Clément Rambour, Nicolas Audebert, Elise Koeniguer, Bertrand Le Saux, Michel Crucianu, Mihai Datcu|2020|[**IEEE Dataport**](https://dx.doi.org/10.21227/w6xz-s898)|
|**Description**|SEN12-FLOOD Dataset Description|ClmRmb|2020|[**GitHub Repository**](https://github.com/ClmRmb/SEN12-FLOOD)|
|**Repository**|Flood Detection in Satellite Images|KonstantinosF|2021|[**GitHub Repository**](https://github.com/KonstantinosF/Flood-Detection---Satellite-Images)|
|**Research Paper**|Flood Detection in Time Series of Optical and SAR Images|C. Rambour, N. Audebert, E. Koeniguer, B. Le Saux, M. Datcu|2020|[**ISPRS Archives**](https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1343/2020/isprs-archives-XLIII-B2-2020-1343-2020.pdf)|

