# Workflow CI

Repository terpisah untuk Kriteria 3. Folder `MLProject/` merupakan MLflow
Project yang dapat menjalankan retraining dari dataset preprocessing.

Jalankan secara lokal:

```powershell
$trackingDirectory = New-Item -ItemType Directory -Force ".\MLProject\mlruns"
$env:MLFLOW_TRACKING_URI = ([System.Uri]$trackingDirectory.FullName).AbsoluteUri
mlflow run .\MLProject --env-manager=local --experiment-name=breast-cancer-ci
```

Workflow GitHub Actions akan:

1. menyiapkan Python 3.12;
2. menjalankan MLflow Project;
3. mengunggah `mlruns` sebagai artefak;
4. membangun dan mendorong Docker image jika secret Docker Hub tersedia.
