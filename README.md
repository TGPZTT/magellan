# Magellan
Automated Flood Detection based on Satellite Images - BME

<h4>Team name:</h4> 
<p>Magellan</p>

<ul>
    <h4>Members:</h4>
    <li>Tóth Ádám László - TK6NT3</li> 
    <li>Szladek Máté Nándor - TGPZTT</li>
</ul>
<br>
<p>
    <h4>Description:</h4> 
    This project focuses on flood detection using only the SEN2 data from the SEN12-FLOOD dataset. The goal is to process Sentinel-2 satellite images, specifically four spectral bands: Band2 (blue), Band3 (green), Band4 (red), and Band8 (infrared). These bands are important for distinguishing between land, water, and other surface features. The process involves organizing the data, checking for empty images, and stacking the bands together for further analysis, which will be used for flood detection tasks.
</p>

<p>
    <h4>Files and folders:</h4>
    The repository contains two Docker Hub links due to the size of the dataset. The two repos contain a before and an after stage of the data preparation.<br>
    Images are based on Linux OS and contain the script file of the data preparation (Data_Preparation.ipynb) and a folder with numerous folders in it for the separated images.<br>
    The image with "raw" code contains the whole dataset, but the one without the "raw" tag contains only the files needed for training. Empty and unnecessary files are removed, and new files were created. The structure of the folders (tree) is different after the completed script operations.<br>
    <br>
    Data is not accessible without authentication, and no in-code authentication is enabled. Data can be accessed after free registration (without any permission), but due to the free accessibility, data is inside the image file.<br>
    <b>Docker Hub (public) links:</b><br>
    <ul>
        <li><a href="https://hub.docker.com/repository/docker/tgpztt/magellan_raw/general">Magellan_RAW</a></li>
        <li><a href="https://hub.docker.com/repository/docker/tgpztt/magellan/general">Magellan</a></li>
    </ul>
</p>
<h4>How to Run the Project</h4>
<p>To run the project, follow these steps:</p>
<ol>
    <li>
        <strong>Pull the Docker Image:</strong>
        <p>Use the following command to pull the desired Docker image from Docker Hub:</p>
        <pre><code>docker pull tgpztt/magellan_raw:latest</code></pre>
    </li>
    <li>
        <strong>Run the Docker Container:</strong>
        <p>Execute the following command to run the Docker image, ensuring to map port <code>8888</code> for Jupyter Notebook access:</p>
        <pre><code>docker run -p 8888:8888 tgpztt/magellan_raw:latest</code></pre>
    </li>
    <li>
        <strong>Access Jupyter Notebook:</strong>
        <p>Once the container is running, open your web browser and navigate to:</p>
        <pre><code>http://localhost:8888</code></pre>
        <p>You should see the Jupyter Notebook interface.</p>
    </li>
    <li>
        <strong>Open and Run the Notebook:</strong>
        <p>In the Jupyter Notebook interface, locate and open the <code>Data_Preparation.ipynb</code> file. You can now run the cells in the notebook, as all required dependencies are pre-installed in the Docker image.</p>
    </li>
</ol>
<p>
<table border="1" cellpadding="10">
    <tr>
        <th>Component</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><strong>Programming Language</strong></td>
        <td>Python 3.11.5</td>
    </tr>
    <tr>
        <td><strong>Jupyter Notebook</strong></td>
        <td>Jupyter Notebook is used for interactive data processing and analysis. It runs on port <code>8888</code>.</td>
    </tr>
    <tr>
        <td><strong>Dependencies</strong></td>
        <td>
            <ul>
                <li><code>jupyterlab</code>: Interactive environment for notebook development</li>
                <li><code>rasterio</code>: Geospatial raster data processing</li>
                <li><code>pandas</code>: Data manipulation and analysis</li>
                <li><code>opencv-python</code>: Image and video processing</li>
                <li><code>notebook</code>: Required to run Jupyter Notebooks</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><strong>System Libraries for OpenCV</strong></td>
        <td>Required libraries for OpenCV to function correctly:
            <ul>
                <li><code>libgl1-mesa-glx</code> (provides OpenGL support)</li>
                <li><code>libglib2.0-0</code> (for image handling support)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><strong>Application Purpose</strong></td>
        <td>This project involves geospatial data processing and analysis, with dependencies for raster data manipulation and image processing.</td>
    </tr>
    <tr>
        <td><strong>Project Access</strong></td>
        <td>The Jupyter Notebook environment is accessed through a web browser on <code>http://localhost:8888</code>.</td>
    </tr>
    <tr>
        <td><strong>Warning</strong></td>
        <td><strong>Do not install any additional software or dependencies</strong> within the environment, as everything needed is already pre-installed in the Docker image.</td>
    </tr>
</table>

</p>
<br>
<p>
 <hr>
<h4>Related Works and Papers:</h4>

<table border="1" cellpadding="10" style="width:100%;">
    <thead>
        <tr>
            <th>Type</th>
            <th>Title</th>
            <th>Authors</th>
            <th>Year</th>
            <th>Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Dataset</td>
            <td>SEN12-FLOOD: a SAR and Multispectral Dataset for Flood Detection</td>
            <td>Clément Rambour, Nicolas Audebert, Elise Koeniguer, Bertrand Le Saux, Michel Crucianu, Mihai Datcu</td>
            <td>2020</td>
            <td><a href="https://dx.doi.org/10.21227/w6xz-s898">IEEE Dataport</a></td>
        </tr>
        <tr>
            <td>Description</td>
            <td>SEN12-FLOOD Dataset Description</td>
            <td>ClmRmb</td>
            <td>2020</td>
            <td><a href="https://github.com/ClmRmb/SEN12-FLOOD">GitHub Repository</a></td>
        </tr>
        <tr>
            <td>Repository</td>
            <td>Flood Detection in Satellite Images</td>
            <td>KonstantinosF</td>
            <td>2021</td>
            <td><a href="https://github.com/KonstantinosF/Flood-Detection---Satellite-Images">GitHub Repository</a></td>
        </tr>
        <tr>
            <td>Research Paper</td>
            <td>Flood Detection in Time Series of Optical and SAR Images</td>
            <td>C. Rambour, N. Audebert, E. Koeniguer, B. Le Saux, M. Datcu</td>
            <td>2020</td>
            <td><a href="https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1343/2020/isprs-archives-XLIII-B2-2020-1343-2020.pdf">ISPRS Archives</a></td>
        </tr>
    </tbody>
</table>
</p>
