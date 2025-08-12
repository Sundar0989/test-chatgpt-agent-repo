# rapid-modeler



### Requirements:
1) Docker 
2) `automl_pyspark/gcp-creds/service_account_key.json`

### Run with Docker (simple)  
1) `sh run.sh`

### Run with Docker  
1) `docker build . -t rapid_modeler`
2) `docker run -p 8080:8080 -v ./automl_pyspark/gcp-creds/service_account_key.json:/root/.config/gcloud/key.json rapid_modeler:latest`
3) Visit http://localhost:8080




