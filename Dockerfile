FROM python:3.11.5

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    rasterio \
    pandas \
    opencv-python \
    numpy \
    matplotlib \
    torch \
    torchvision \
    tifffile \
    tensorflow

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
