# Magellan
**Automated Flood Detection Using Satellite Images**  
**Budapest University of Technology and Economics (BME)**

## Team Name:
**Magellan**

## Team Members:
- **Tóth Ádám László** (ID: TK6NT3)
- **Szladek Máté Nándor** (ID: TGPZTT)

 The main goal of this project is to detect flooded areas using satellite images from the SEN12-FLOOD dataset, which comes from the Sentinel-2 satellite. Flood detection is really important because it helps reduce damage from natural disasters and allows for better early warning systems. Sentinel-2’s multispectral images contain detailed information about land surfaces, vegetation, and water, but these images can be tricky to analyze because they’re so complex. By focusing on just a few of the satellite’s spectral bands (Blue, Green, Red, and NIR), we can simplify the problem and more easily pinpoint flooded regions. The NIR (Near-Infrared) channel is useful because it can pass through thin clouds and is good at detecting plants and water. This makes it helpful for things like flood detection and classifying different types of land.

<hr>

## Related Files

| **Related Files**                                    | **Source**                                                                         |
|------------------------------------------------------|------------------------------------------------------------------------------------|
| Documentation                                   | In this repository                                                                |
| Magellan.ipynb                                       | In this repository                                                                |
| SEN12FLOOD.rar                                      | Google Drive: [SEN12FLOOD.rar](https://drive.google.com/file/d/1F5HYMFQyy5EfpAvPDDSg0cG1lVdm1O2y/view?usp=sharing)  |
| PyTorch model (flood_model.tph)                      | Google Drive: [flood_model.tph](https://drive.google.com/file/d/12XxgFJ3EUggyMby4KaEDLtvDO0KcpJus/view?usp=sharing) |
| Keras model (magellan_model_keras.keras)              | Google Drive: [magellan_model_keras.keras](https://drive.google.com/file/d/1-qDnHM7KCe3EIREtG4u-oWiA5B_ivy1N/view?usp=sharing) |
| Pytorch classification results (pytorch_classification_results.csv)              | In this repository |
| Keras classification results (keras_classification_results.csv)              | In this repository |
| Docker image (1st milestone_RAW)                      | DockerHub: [1st milestone_RAW](https://hub.docker.com/repository/docker/tgpztt/magellan_raw/general) |
| Docker image (1st milestone_processed)                | DockerHub: [1st milestone_processed](https://hub.docker.com/repository/docker/tgpztt/magellan/general) |
| Docker image (2nd milestone)                          | DockerHub: [2nd milestone](https://hub.docker.com/repository/docker/tgpztt/magellan_milestone2) |
| Docker image (Final)                                  | DockerHub: [final](https://hub.docker.com/repository/docker/tgpztt/magellan_milestone2)                                                                    |

## Data Processing and Model Development

### 1. Data Quality and Challenges
The quality of the processed data is a critical factor in determining model performance. The SEN12-FLOOD database, while comprehensive, presents challenges such as blurred flood contours and significant noise, which complicate the task of distinguishing flooded areas. This project focuses exclusively on Sentinel-2 optical data due to its rich detail on land surfaces, vegetation, and water. However, the inherent complexity of this data makes it difficult for models to extract meaningful features. 

### 2. Developing the Baseline Model
The baseline model, built using Keras, provided a strong foundation for the project by illustrating core flood detection principles and serving as a reference for further development. With its simpler architecture, the baseline model allowed us to:
- Understand the data.
- Measure the effects of data augmentation.
- Identify key directions for model refinement.
After upgrading this model achieved robust results, outperforming the more complex PyTorch model in metrics such as accuracy and precision. Its performance highlights that simplicity, paired with iterative improvements, can often yield better results than more intricate designs.

### 3. Advanced Model with more consiferations
The advanced model, implemented in PyTorch, aimed to improve flood detection through modern techniques such as input normalization, class balancing, and dropout regularization. Despite these enhancements, its performance fell short compared to the baseline model. This outcome underscores the importance of fine-tuning and data quality in achieving effective results. The PyTorch model’s complexity may have introduced challenges in fully leveraging the dataset within the time constraints of this project. Nonetheless, with better data quality, optimized hyperparameters, and additional refinement, the advanced model holds significant potential for superior performance in the future.


### Key Evaluation Steps and Metrics

| **Evaluation Steps**             | **Details**                                                                                                                                                                                                                                      |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Spectral Band Statistics**     | - Mean and Normalized Standard Deviation (Mean_Std).<br> - Standard Deviation (Std_Dev) and Normalized Standard Deviation (Std_Dev_Std).<br> - Mean Contribution (Mean_Contribution).<br> - Standard Deviation-Based Contribution (Std_Contribution). |
| **Class Distribution Analysis**  | - Distribution of data classes: CLEAR and FLOODED.<br> - Class balance assessment with sample proportions.                                                                                                                                       |
| **Visualizations**               | - Pie Chart: Spectral band contributions (Blue, Green, Red, NIR).<br> - Confusion Matrix: Classification performance.<br> - ROC Curve: Classification evaluation.<br> - PR Curve: Precision and sensitivity comparison.                          |
| **Performance Metrics**          | - Accuracy<br> - Precision<br> - Recall<br> - F1-score<br> - Class-specific summaries                                                                                                                                                            |
| **Relative Contribution Analysis** | - Quantification of average and standard deviation contributions for each spectral band.<br> - Identification of dominant bands (e.g., NIR).                                                                                                     |

### Additional Features
- **Wandb Integration**: Logs metrics and tracks model performance across epochs.
- **TensorBoard Integration**: Visualizes training progress and metrics for the Keras model in real time.
- **Gradio Interface**: Allows testing the model on uploaded files with prediction results.
  
In this project, we leveraged **Wandb** for logging metrics, **TensorBoard** for visualizing training progress, and **Gradio** for an intuitive user interface, allowing for easy interaction with the model.

### Model Evaluation and Comparison
The project explores two models for flood detection:

1. **Solution 1 (Keras Model)**: A simple CNN-based model using TensorFlow/Keras with integrated TensorBoard for real-time progress tracking.
2. **Solution 2 (PyTorch Model)**: A more complex model using PyTorch Lightning with Wandb for logging metrics.

Despite the additional complexity of the PyTorch model, the Keras model performed better, highlighting the importance of simplicity in certain cases.

### Conclusion
The goal of this project was to first define a strong baseline model for flood detection using Sentinel-2 imagery and then build a more advanced model to improve upon it. Our initial baseline, built with Keras, evolved into a highly effective solution as we refined it through further tuning and data engineering. Meanwhile, we designed a more complex PyTorch model incorporating advanced techniques such as class balancing, normalization, and dropout. Although the PyTorch model did not outperform the improved Keras baseline, this result highlights potential areas for further exploration rather than the limitations of the approach itself.
The Keras model achieved better results across all key metrics, including accuracy, precision, and AUC (0.77 vs. 0.50), demonstrating that simplicity paired with iterative improvement can deliver strong performance. On the other hand, the PyTorch model has significant untapped potential. With more time and resources, hyperparameter optimization, better normalization strategies, and advanced data augmentation could improve its performance. Additionally, the quality and characteristics of the Sentinel-2 visual data play a critical role in training both models. If the data is inherently noisy or lacks sufficient representation of flooding, it could limit the model's ability to generalize. Exploring higher-quality datasets or supplementing with radar data could unlock the full potential of the PyTorch model.
While the PyTorch model struggled in this case, the Keras model provided a reliable and practical solution for flood detection, demonstrating that simplicity and careful design can often deliver strong results. This project underscores the importance of understanding both the data and the models in use, and the lessons learned here offer valuable directions for future work, including improving data quality and further optimizing advanced model architectures.



<hr>

### How to run
The repository contains four Docker Hub links:

The first one describes data processing. The first one shows the first one, which contains the original data. The first one is the original data. Due to the size of the data, the other DockerHub images contain only the processed, carefully selected data.

Data is not accessible without authentication, and no in-code authentication is enabled. Data can be accessed after free registration (without any permission), but due to the free accessibility, data is inside the image file.

The file operations are platform independent, so the image file can be downloaded and run on a windows system without any permission. 



# Project Instructions - How to Run the Project

| **Step** | **Title**                  | **Instructions** |
|----------|----------------------------|-------------------|
| **A**    | **Docker Image (Cross-Platform)** | 1. **Pull the Docker Image**:<br>Use the following command to pull the required Docker image from Docker Hub:<br><br>`docker pull tgpztt/magellan_final:latest`<br><br>2. **Run the Docker Container**:<br>Execute the following command to run the Docker image, ensuring to map port 8888 for Jupyter Notebook access:<br><br>`docker run -p 8888:8888 tgpztt/magellan_final:latest`<br><br>3. **Access Jupyter Notebook**:<br>Once the container is running, open your browser and navigate to:<br><br>[http://localhost:8888](http://localhost:8888)<br><br>If prompted for a token, copy it from the terminal output. |
| **B**    | **Data Preparation**       | 1. **Download the Dataset**:<br>Obtain the dataset from the IEEE website (link is in the "Related Work and Papers" section).<br><br>2. **Obtain the Notebook**:<br>Download the `magellan.ipynb` file from this repository.<br><br>3. **Organize Files**:<br>Ensure that the dataset and the `.ipynb` file are in the same folder.<br><br>4. **Run Preprocessing**:<br>Open `magellan.ipynb` and execute the initial cells to preprocess the dataset. Detailed instructions are provided in the markdown cells. |
| **C**    | **For Training**           | 1. **Download the Dataset**:<br>Ensure the `SEN12FLOOD` folder (containing subfolders and JSON files) is prepared.<br><br>2. **Download the Notebook**:<br>Obtain `magellan.ipynb` from this repository.<br><br>3. **Organize Files**:<br>Place the `SEN12FLOOD` folder and the `.ipynb` file in the same directory.<br><br>4. **Run Training**:<br>Open `magellan.ipynb` and execute the block under the **"Data loading"**, the **"Solution 1"** or the **'Solution 2'** section. Instructions are in the markdown cells. |
| **D**    | **For Evaluation**         | 1. **Download Pre-Trained Models**:<br>Obtain the following models from this repository:<br> - `flood_model.pth`<br> - `magellan_model_keras.keras`<br><br>2. **Download the Notebook**:<br>Obtain `magellan.ipynb` from this repository.<br><br>3. **Download Classification Results**:<br>Obtain the classification results files:<br> - **PyTorch classification results**: `pytorch_classification_results.csv`<br> - **Keras classification results**: `keras_classification_results.csv`<br><br>4. **Organize Files**:<br>Ensure the following files are placed in the same folder:<br> - `magellan.ipynb`<br> - `pytorch_classification_results.csv`<br> - `keras_classification_results.csv`<br> - `flood_model.pth`<br> - `magellan_model_keras.keras`<br><br>5. **Run Evaluation**:<br>Open `magellan.ipynb` and execute the blocks under the **"Data loading"**, the **"Evaluating the Keras Model"** or the **'Evaluating the PyTorch Model'** section. Detailed instructions are provided in the markdown cells. |





<hr>

## Running Environment and Components

| Component                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                 |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Programming Language**       | Python 3.11.5                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Jupyter Notebook**           | Jupyter Notebook is used for interactive data processing and analysis. If you have pulled the docker file it runs on port 8888.                                                                                                                                                                                                                                                                                               |
| **Dependencies**               | **_File and Directory Management_**<br> - `os` - File and directory operations.<br> - `shutil` - File manipulation (e.g., copy, move, delete).<br> - `json` - Reading and writing JSON files.<br><br> **_Data Manipulation_**<br> - `numpy` - Numerical computations and array manipulation.<br> - `pandas` - Dataframe and tabular data handling.<br> - `random` - Random number generation.<br><br> **_Image Handling and Processing_**<br> - `cv2` - Image processing and computer vision.<br> - `Pillow [Image]` - Opening and manipulating image files.<br> - `tifffile` - Handling TIFF image files.<br><br> **_Visualization_**<br> - `matplotlib.pyplot` - General-purpose plotting and visualization.<br> - `seaborn` - Statistical data visualization.<br><br> **_Deep Learning (PyTorch Ecosystem)_**<br> - `torch` - Core library for tensors and models.<br> - `torch.nn.functional [F]` - Functional utilities for neural networks.<br> - `torchvision` - Pre-trained models, datasets, and transforms.<br> - `torch.utils.data [WeightedRandomSampler]` - Weighted sampling for datasets.<br> - `torchmetrics` - Metrics for model evaluation.<br> - `pytorch_lightning` - High-level PyTorch training abstraction.<br> - `pytorch_lightning.callbacks [EarlyStopping]` - Early stopping callback.<br> - `pytorch_lightning [Trainer]` - Manages PyTorch training loops.<br><br> **_Deep Learning (TensorFlow and Keras Ecosystem)_**<br> - `tensorflow` - Core library for deep learning.<br> - `tensorflow.keras.models [Sequential]` - Model building.<br> - `tensorflow.keras.layers` - Includes `[Dense]`, `[Conv2D]`, `[MaxPool2D]`, `[Flatten]`, and `[Dropout]` for neural network layers.<br> - `tensorflow.keras.preprocessing.image [ImageDataGenerator]` - Data augmentation.<br> - `tensorflow.keras.optimizers [Adam]` - Adam optimizer.<br> - `tensorflow.keras.utils [plot_model]` - Visualize model architecture.<br><br> **_Machine Learning Metrics_**<br> - `sklearn.metrics` - Includes `[classification_report]`, `[confusion_matrix]`, `[roc_curve]`, `[auc]`, `[precision_recall_curve]`, and `[average_precision_score]` for evaluating model performance.<br><br> **_Experiment Tracking and Interface_**<br> - `wandb` - Experiment tracking and visualization.<br> - `gradio` - Build user interfaces for machine learning models.<br> - `tensorboard` - For visualizing and monitoring model training and performance during TensorFlow and PyTorch model training. |
| **Application Purpose**        | This project involves geospatial data processing and flood detection, relying on raster data manipulation, image augmentation, and a baseline model for flood classification.                                                                                                                                                                                                                                                   |
| **Project Access**             | If you have pulled a Docker image, the Jupyter Notebook environment is accessed through a web browser at [http://localhost:8888](http://localhost:8888).                                                                                                                                                                                                                                                                       |
| **Warning**                    | **Do not install any additional software or dependencies** within the environment, as everything needed is already pre-installed in the Docker image.                                                                                                                                                                                                                                                                         |



<hr>

**Related Works and Papers:**

|**Type**|**Title**|**Authors**|**Year**|**Link**|
| :- | :- | :- | :- | :- |
|**Dataset**|SEN12-FLOOD: a SAR and Multispectral Dataset for Flood Detection|Clément Rambour, Nicolas Audebert, Elise Koeniguer, Bertrand Le Saux, Michel Crucianu, Mihai Datcu|2020|[**IEEE Dataport**](https://dx.doi.org/10.21227/w6xz-s898)|
|**Description**|SEN12-FLOOD Dataset Description|ClmRmb|2020|[**GitHub Repository**](https://github.com/ClmRmb/SEN12-FLOOD)|
|**Repository**|Flood Detection in Satellite Images|KonstantinosF|2021|[**GitHub Repository**](https://github.com/KonstantinosF/Flood-Detection---Satellite-Images)|
|**Research Paper**|Flood Detection in Time Series of Optical and SAR Images|C. Rambour, N. Audebert, E. Koeniguer, B. Le Saux, M. Datcu|2020|[**ISPRS Archives**](https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1343/2020/isprs-archives-XLIII-B2-2020-1343-2020.pdf)|

