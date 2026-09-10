# Free real-model source

The app can run `VaigandlaHemanth/leaf-disease-clip-vit` locally without an API key or paid service. It is a PlantVillage classifier with 38 classes across 14 crops.

- Model: https://huggingface.co/VaigandlaHemanth/leaf-disease-clip-vit
- Dataset: https://plantvillage.psu.edu/

## Enable locally on Windows

```bat
venv\Scripts\activate
pip install -r requirements-ml.txt
```

In `.env`, set `USE_LOCAL_MODEL=true`, then run `python app.py`. The first start downloads the public model, so keep internet connected. It can take several minutes and needs several GB of available disk/RAM after ML packages install.

Render's free service is too small for dependable large vision-model inference, so it remains a UI demo. Do not treat PlantVillage test results as a guarantee for outdoor field photos.
