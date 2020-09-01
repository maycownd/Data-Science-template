import requests
import json

checkdeploymentError = ""


def checkdeployment():
    # function which tests whether the deployment is successful
    try:
        # Creating a sample data for fashion MNIST (28*28 images)
        sample_data = [0 for k in range(784)]

        headers = {'Content-Type': 'application/json'}
        r = requests.post(url="http://127.0.0.1:5000/api",
                          headers=headers, json={"data": sample_data})
        r.json()
        result = (type(r.json()[0]) == int)
    except ConnectionRefusedError:
        result = False
    return result
