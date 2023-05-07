<h1> traffic-sign-recognition </h1>

</br>
<h3> Description </h3>
Traffic sign recongnition is a system used in self driving cars and driving assistance systems for automatic detection of various traffic signs and act or guide according to the sign</br>
This system will assistant humans when then exceed any traffic limit in driving assistance systems </br>
This system directs the car according to the traffic sign in the self driving cars </br>
In this project, we used deep neural networks - convolutional neural network  for developing traffic sign recognition system </br>
The model accuracy is 0.97  </br> 
We used German dataset which consists around 50,000 images and 43 classes for developing this model </br>
</br>
<h3> Objectives </h3>
In this project, we are building a neural network model for the German Dataset which detects the particular image type and gives the description of the image i.e, about anyone of the traffic sign </br>
In the real-time implementation of this model, it predicts the traffic sign with high accuracy </br>
we are developing a web interface for the model </br>
</br>
<h3> Scope </h3>
This model can help self-driving cars to recognize traffic signs </br>
The model will assist the drivers by recognizing the traffic sign </br>
Traffic sign recognition has become an important research topic in the area of convolution neural network </br>
In the future, we will implement the traffic sign capture and  recognition over video datasets </br>
</br> 
<h3> Technologies and libraries </h3>
Python </br>
Numpy </br>
Pandas </br>
mathplot </br>
OpenCv </br>
Scikit-Learn </br>
TensorFlow </br>
Keras </br>
gTTS </br>
Translate </br>
PlaySound </br>
PIL </br>
</br>
<h3> Environment Required </h3>
Must create new environment with Tensorflow and Keras installation in anaconda </br>
Use same Tensorflow environment in terminal to run app in local machine </br>
Create Flask environemnt to run our web.py python app and its launching page index.html </br>

<h3> Features </h3>
it classifies a traffic sign image from the 43 class types </br>
it classifies the image and play the audio to the meaning of traffic sign </br>
will classify the meaning and translate the meaning of traffic sign to preferred language </br>
<h3> My Contribution to the Project </h3>
<b>As part of the data preparation process for the traffic sign recognition project, I performed data preprocessing which involved normalizing, reshaping, and one-hot encoding the input images. The normalization step ensured that all images had a consistent brightness and contrast, while the reshaping step resized all images to a fixed size. Finally, I applied one-hot encoding to convert the categorical target labels into a binary vector representation, which is required for machine learning algorithms to process the data.</b></br>
<b>Data preprocessing:</b></br>
"Data preprocessing is an essential step in any machine learning project, as it can significantly impact the accuracy and performance of the model. In the context of traffic sign recognition, the preprocessing steps involve normalizing, reshaping, and one-hot encoding the images.</br>

<b>Normalization:</b></br>
The first step in data preprocessing is normalization, which involves scaling the pixel values of the input images to a common range. This ensures that all images have similar brightness and contrast, which can help the model learn better features. Typically, image normalization involves subtracting the mean pixel value and dividing by the standard deviation.</br>

<b>Reshaping:</b></br>
The second step is to reshape the images to a common size, as traffic sign images can vary in size and shape. This can be achieved by cropping, padding, or resizing the images to a fixed size. </br>
<b>One-Hot Encoding:</b></br>
The third step is one-hot encoding the target labels. This is necessary because machine learning algorithms typically require numeric values as input. In traffic sign recognition, the target labels are categorical, meaning they belong to a specific class or category. One-hot encoding converts the categorical labels into a binary vector representation, where each element in the vector corresponds to a different class. </br>

