# magellan
Automated Flood Detection based on Satellite Images - BME

<h4>Team name: </h4> Magellan<br>

<ul>
 <h4> Members: </h4>
  <li>Tóth Ádám László - TK6NT3</li> 
  <li>Szladek Máté Nándor - TGPZTT</li>
</ul>
<br>
<p>
<h4>Description:</h4> 
  This project focuses on flood detection using only the SEN2 data from the SEN12-FLOOD dataset. The goal is to process Sentinel-2 satellite images, specifically four spectral bands: Band2 (blue), Band3 (green), Band4 (red), and Band8 (infrared). These bands are important for distinguishing between land, water, and other surface features. The process involves organizing the data, checking for empty images, and stacking the bands together for further analysis, which will be used for flood detection tasks.
</p>

<p><h4>Files and folders:</h4>
The repositorie contains two dockerhub links due to the size of the dataset. The two repo contains a before and an after stage of the data preparation.<br>
Images are based on linux os, and contains the scrpt file of the data preparation (Data_Preparation.ipynv) and a folder with numerous folders in it for the seperated images. <br>
The image with "raw" code contains the whole dataset but the one without the "taw" tag contains only the files needed for training. Empty and not needed files are removed but also new files were created. The structure of the folders (tree) is different after the completed script operations. <br>
<br>
Data is not accessable without authentication, and no incode authentication is enabled. Data can be accessed after free registration (without any permission) but due to the free accessability data is inside the image file.<br>
<b>Docker hub (public) links:</b><br>
<ul>
  <li><a href="https://hub.docker.com/repository/docker/tgpztt/magellan_raw/general">Magellan_RAW</a></li>
  <li><a href="https://hub.docker.com/repository/docker/tgpztt/magellan/general">Magellan</a></li>
</ul>
</p>
<h4>How to run it</h4>
<p>
  <ol>
    <li>Pull the selected image file (e.g. docker pull tgpztt/magellan_raw:latest)</li>
    <li>Run the image and map the 8888 port! (e.g. docker run -p 8888:8888 tgpztt/magellan_raw:latest) </li>
    <li>Once the container is running, you should be able to access Jupyter Notebook in your browser at: http://localhost:8888</li>
    <li>Open Data_Preparation.ipynb file and run it. The requirements are installed.</li>
  </ol>
</p>
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
<h4>Related works and papers:</h4><hr>
<ul>
 <i> Dataset:</i>
<li>Clément Rambour, Nicolas Audebert, Elise Koeniguer, Bertrand Le Saux, Michel Crucianu, Mihai Datcu. (2020). SEN12-FLOOD : a SAR and Multispectral Dataset for Flood Detection . IEEE Dataport. https://dx.doi.org/10.21227/w6xz-s898</li><br>
  <i>Description of the dataset (not original):</i>
<li><a href="https://github.com/ClmRmb/SEN12-FLOOD">GitHub - ClmRmb</a></li>
  <hr>
  <i>Repositories:</i>
<li><a href="https://github.com/KonstantinosF/Flood-Detection---Satellite-Images">GitHub - KonstantinosF
</a></li><hr>


<i>Multispectral dataset - advantages:</i>
<li>Flood Detection in Time Series of Optical and SAR Images, C. Rambour,N. Audebert,E. Koeniguer,B. Le Saux, and M. Datcu, ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2020, 1343--1346<a href="https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1343/2020/isprs-archives-XLIII-B2-2020-1343-2020.pdf">PDF</a></li>
  
</ul>
  
</p>
