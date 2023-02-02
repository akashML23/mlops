# mlops-assignment
### Two Approaches
#### 1. Logistic Regression: A multi-class classification using mnist dataset from sklearn is trained with logistic regression. Accuracy is around 90% but this approach gives a very lightweight docker image that can be deployed with minimal hardware configuration.
#### 2. CNN: A convolution neural network is trained using tensorflow. This approach gives higher accuracy but since tensorflow is used, it makes docker image quite high ~2.14 GB.

### Inference Script:
#### Flask is used to create a rest endpoint to serve the model. This rest endpoint expects an image as a body and implicitly loads the model and calls the predict method that returns the result.

### Building Docker Image:
#### Python-slim is the base image that is taken for both the approaches. The docker image sizes are 247MB and 2.14GB respectively. Packaged image is pushed to a docker repository

### K8s deployment:
#### Firstly, a deployment is created to run the above images inside a container.Secondly, a NodePort type service is created with a selected port and attached to the specific deployment.The deployment initiates a pod with run the container having our mnist inference service image. 