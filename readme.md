This project focuses on the processing of Sentinel-2 satellite imagery from the SEN12-FLOOD database and flood detection. Four spectral bands are used for processing: Band2 (blue), Band3 (green), Band4 (red) and Band8 (near-infrared). These bands are of particular importance for distinguishing land, water and other surface features. Throughout the project, special attention was paid to data organisation, filtering out blank images and band alignment to provide a suitable basis for flood detection tasks.

## Data Processing and Model Development
1. **Data quality and challenges**

The quality of the processed data has a significant impact on model performance. The SEN12-FLOOD database, although a comprehensive source, does not always provide sufficient resolution and quality for flood detection. The contours of the flooded areas are often blurred or other noise factors hamper the analysis. As a result, even advanced models have difficulty in accurately clustering individual classes.

1. **Developing the base model**

The basic model we have developed serves as one of the main starting points for the project. The purpose of the basic model is to illustrate the basic operating principle of the flood detection task and to serve as a reference for further development. The basic model, with its simpler architecture, will help to understand the data, measure the impact of augmentations, and determine the direction of parameter tuning.

1. **Advanced model and its limitations**

Following the basic model, a more advanced model has been implemented that uses more modern algorithms and techniques to optimize flood detection. Although the new model is more comprehensive and robust in several aspects, the classification results did not reach the desired level due to the poor quality of the data and the low resolution. This highlights the fact that preliminary data quality control and improvement is a key factor for the project.

As part of the project, several evaluation and data analysis steps were carried out, for which graphs were produced. These are shown in the attached document

**1. Spectral Band Statistics:**

- **Mean** and **normalized standard deviation (Mean\_Std)**.
- **Standard deviation (Std\_Dev)** and **normalized standard deviation (Std\_Dev\_Std)**.
- **Mean contribution (Mean\_Contribution)** to the dataset.
- **Standard deviation-based contribution (Std\_Contribution)**.

**2. Class Distribution and Sample Analysis:**

- Distribution of data classes: CLEAR and FLOODED.
- Class balance assessment: calculation of the proportion of samples per class.

**3. Visualizations:**

- **Pie chart** showing contribution ratios of spectral bands (Blue, Green, Red, NIR).
- **Confusion matrix** illustrating classification performance.
- **ROC curve (Receiver Operating Characteristic)** for evaluating classification performance.
- **AUC (Area Under Curve)**: the area under the ROC curve to assess model accuracy.
- **Precision-Recall curve (PR Curve)** to compare model sensitivity and precision.

**4. Model Performance Metrics:**

- **Accuracy**.
- **Precision**.
- **Recall**.
- **F1-score**.
- Class-specific metrics and summaries.

**5. Relative Contribution Analysis:**

- Quantification of average and standard deviation-based contributions of each spectral band.
- Identification of dominant bands (e.g., NIR).

**Wandb feature is also added but you have to use your own API key to see the telemetrics.**

**Gradio is also added, so you can test the model on a chosen file that you can upload and see the prediction result.** 

### How to run
The repository contains four Docker Hub links:

The first one describes data processing. The first one shows the first one, which contains the original data. The first one is the original data. Due to the size of the data, the other DockerHub images contain only the processed, carefully selected data.

Data is not accessible without authentication, and no in-code authentication is enabled. Data can be accessed after free registration (without any permission), but due to the free accessibility, data is inside the image file.

The file operations are platform independent, so the image file can be downloaded and run on a windows system without any permission. 

DockerHub public links:

- [Magellan_RAW](https://hub.docker.com/repository/docker/tgpztt/magellan_raw/general)
- [Magellan](https://hub.docker.com/repository/docker/tgpztt/magellan/general)
- [Magellan_Milestone2](https://hub.docker.com/repository/docker/tgpztt/magellan_milestone2) (contains baseline model, augmentation, JSON filtering)

How to Run the Project

A: Docker image (cross-platform)

B. Data preparation

C: For training

D: For evaluating

B, C, D steps are using the same magellan.ipynb file. You can run all the tasks above if you have generated or downloaded the files needed. 

If you are using B,C or D method, be sure you have installed the required packages!

|<p>**A.**</p><p></p><p>To run the project, follow these steps. The code part is for the 2th milestone, but can be changed according to the required project. (Docker Hub site contains the needed code)</p><p>1. **Pull the chosen Docker Image:**</p><p>Use the following command to pull the desired Docker image from Docker Hub:</p><p>docker pull tgpztt/magellan\_milestone2:latest</p><p>2. **Run the Docker Container:**</p><p>Execute the following command to run the Docker image, ensuring to map port 8888 for Jupyter Notebook access:</p><p>docker run -p 8888:8888 tgpztt/magellan\_milestone2:latest</p><p>3. **Access Jupyter Notebook:**</p><p>Once the container is running, open your web browser and navigate to:</p><p>http://localhost:8888</p><p>You should see the Jupyter Notebook interface.</p><p>4. **Open and Run the Notebook:**</p><p>In the Jupyter Notebook interface, locate and open the Magellan.ipynb file. You can now run the cells in the notebook, as all required dependencies are pre-installed in the Docker image.</p><p></p>|
| :- |
|<p>**B.**</p><p></p><p>Download the dataset from IEEE website (link is in the Related Work and Papers section)</p><p></p><p>Download final.ipynb file from this repository</p><p></p><p>The ipynb file and the downloaded dataset should be in the same folder!</p><p></p><p>Open final.ipynb and run the first blocks. The needed markdown is added, so you will know what to run.</p>|
|<p>**C.**</p><p></p><p>Download the prepared dataset from this repository.</p><p>Download final.ipynb file from this repository</p><p>The ipynb file and SEN12FLOOD folder (containing 2037 subfolder and json files) should be in the same folder!</p><p></p><p>Open final.ipynb and run the block after the „Model” title. The needed markdown is added, so you will know what to run.</p><p></p>|
|<p>**D.** </p><p></p><p>Download the model file (flood\_model.pth) from this repository.</p><p>Download final.ipynb file from this repository</p><p>The ipynb file and the model file should be in the same folder!</p><p></p><p>Open final.ipynb and run the block after the „Eval” title. The needed markdown is added, so you will know what to run.</p><p></p>|

## Running environment and components

|**Component**|**Description**|
| :- | :- |
|**Programming Language**|Python 3.11.5|
|**Jupyter Notebook**|Jupyter Notebook is used for interactive data processing and analysis. If you have pulled the docker file it runs on port 8888.|
|**Dependencies**|<p>**File and Directory Management**</p><p>- **os** - File and directory operations.</p><p>- **shutil** - File manipulation (e.g., copy, move, delete).</p><p>- **json** - Reading and writing JSON files.</p><p>-----</p><p>**Data Manipulation**</p><p>- **numpy** - Numerical computations and array manipulation.</p><p>- **pandas** - Dataframe and tabular data handling.</p><p>- **random** - Random number generation.</p><p>-----</p><p>**Image Handling and Processing**</p><p>- **cv2** - Image processing and computer vision.</p><p>- **Pillow [Image]** - Opening and manipulating image files.</p><p>- **tifffile** - Handling TIFF image files.</p><p>-----</p><p>**Visualization**</p><p>- **matplotlib.pyplot** - General-purpose plotting and visualization.</p><p>- **seaborn** - Statistical data visualization.</p><p>-----</p><p>**Deep Learning (PyTorch Ecosystem)**</p><p>- **torch** - Core library for tensors and models.</p><p>- **torch.nn.functional [F]** - Functional utilities for neural networks.</p><p>- **torchvision** - Pre-trained models, datasets, and transforms.</p><p>- **torch.utils.data [WeightedRandomSampler]** - Weighted sampling for datasets.</p><p>- **torchmetrics** - Metrics for model evaluation.</p><p>- **pytorch\_lightning** - High-level PyTorch training abstraction.</p><p>- **pytorch\_lightning.callbacks [EarlyStopping]** - Early stopping callback.</p><p>- **pytorch\_lightning [Trainer]** - Manages PyTorch training loops.</p><p>-----</p><p>**Deep Learning (TensorFlow and Keras Ecosystem)**</p><p>- **tensorflow** - Core library for deep learning.</p><p>- **tensorflow.keras.models [Sequential]** - Model building.</p><p>- **tensorflow.keras.layers** - Includes [Dense], [Conv2D], [MaxPool2D], [Flatten], and [Dropout] for neural network layers.</p><p>- **tensorflow.keras.preprocessing.image [ImageDataGenerator]** - Data augmentation.</p><p>- **tensorflow.keras.optimizers [Adam]** - Adam optimizer.</p><p>- **tensorflow.keras.utils [plot\_model]** - Visualize model architecture.</p><p>-----</p><p>**Machine Learning Metrics**</p><p>- **sklearn.metrics** - Includes [classification\_report], [confusion\_matrix], [roc\_curve], [auc], [precision\_recall\_curve], and [average\_precision\_score] for evaluating model performance.</p><p>-----</p><p>**Experiment Tracking and Interface**</p><p>- **wandb** - Experiment tracking and visualization.</p><p>- **gradio** - Build user interfaces for machine learning models.</p><p></p>|
|**Application Purpose**|This project involves geospatial data processing and flood detection, relying on raster data manipulation, image augmentation, and a baseline model for flood classification.|
|**Project Access**|If you have pulled a Docker image, the Jupyter Notebook environment is accessed through a web browser at http://localhost:8888.|
|**Warning**|**Do not install any additional software or dependencies** within the environment, as everything needed is already pre-installed in the Docker image.|


**Related Works and Papers:**

|**Type**|**Title**|**Authors**|**Year**|**Link**|
| :- | :- | :- | :- | :- |
|**Dataset**|SEN12-FLOOD: a SAR and Multispectral Dataset for Flood Detection|Clément Rambour, Nicolas Audebert, Elise Koeniguer, Bertrand Le Saux, Michel Crucianu, Mihai Datcu|2020|[**IEEE Dataport**](https://dx.doi.org/10.21227/w6xz-s898)|
|**Description**|SEN12-FLOOD Dataset Description|ClmRmb|2020|[**GitHub Repository**](https://github.com/ClmRmb/SEN12-FLOOD)|
|**Repository**|Flood Detection in Satellite Images|KonstantinosF|2021|[**GitHub Repository**](https://github.com/KonstantinosF/Flood-Detection---Satellite-Images)|
|**Research Paper**|Flood Detection in Time Series of Optical and SAR Images|C. Rambour, N. Audebert, E. Koeniguer, B. Le Saux, M. Datcu|2020|[**ISPRS Archives**](https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1343/2020/isprs-archives-XLIII-B2-2020-1343-2020.pdf)|


