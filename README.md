# Malaria_Catcher
A Computer Vision project that is fed a certain type of cell image and returns whether or not the cell is infected with Malaria. 

The Project is currently deployed via Google Cloud Functions. 
***To try it out you can send a POST request to the following URL via an API-Platform like Postman.***
https://europe-west6-meta-iterator-337819.cloudfunctions.net/function-get_predict

## Access via API-Platform: 
- set the type of request from *"GET"* to ***"POST"***
- Go to *"Body"*
- add a *KEY* and name it **"file"**
- set the type of *KEY* as *"File"*
- **select the file** you want to have evaluated in *"Value"*
- *Send*

## Files included in the repository:
- ***Images*** from a different dataset that contains images of a similar style to those with which the model was trained. You can use them in your POST requests. 
- The called ***function*** in GCP 
- An ***api script*** that runs a local server for testing purposes before deployment to GCP. 
The server runs on **localhost:8000/predict**. You can also reach it via Postman. 
- ***Models*** used for the evaluation. The one currently used is **"v2_e50.h5"**. It's also available in the standard folder format.
- Jupyter Notebook of the ***model training*** process.
