# Reverse-Visual-Search
####PROBLEM STATEMENT:
We have all played the game of “spot the difference” in which we need to find differences between two similar images. To build upon the context, can you find images that are similar to a given image? The google reverse image search is an apt description of what we are going to build today. Our problem statement is to find N similar images, given an input image.  

####INTRODUCTION:
As humans, we are fairly good at comparing an image and finding similar images just by looking at an image. But what if we need to find similar images from a pool of thousands of images? This is where our machines come into the picture. To be a bit more specific, we can utilize the concepts from computer vision, and deep learning to build a model that can implement this task for us in a matter of minutes, if not days. 
Unlike the other machine learning project that is based on metadata such as keywords, tags, or description, we are going to build our project on the ‘content based’ which means that we are going to extract meaningful features from the images to achieve our goal of finding similar images. It is a common scenario in a real-life-based scenario in which we are only given an image and we do not have the corresponding meta-data associated with it. Fortunately, the recent advances in the field of computer vision and deep learning will help us despite this shortcoming. 
DATA
In this project, we are going to utilize the LFW Dataset (Labelled Faces in the Wild), a compilation of face pairs of positively and negatively weighted images samples collected by the researchers of the University of Massachusetts. The dataset comprises 13233 images of 5749 people with 1680 people having two or more images. LFW Dataset is the benchmark for pair matching, also known as face verification. For our project, we are going to train our model on this dataset and later utilize this model for a reverse image search on a given face image. 

<p align="center">
  <img src="(https://user-images.githubusercontent.com/34485564/165446646-7d839b67-fe85-4cbc-84ab-fd88576510ea.png)" />
</p>

####PROCEDURE
This section discusses the Facial recognition models that have been used throughout the project. 

<p align="center">
  <img src="(https://user-images.githubusercontent.com/34485564/165446704-4d1045fa-39bd-430a-82a7-7562134f90b3.png)" />
</p>

Figure 1 BaseLine Architecture

The project is divided into three parts as follows:
1.	Creating a baseline performance: Use ResNet50 for finding the embeddings of an image and providing it as an input to our machine learning model, in our case K-Nearest Neighbor.
2.	Initial Improvement on Baseline Performance: Utilizing MTCNN Face detection to extract target images (faces) and utilizing ResNet50 along with K- Nearest Neighbor (similar to the baseline model).
3.	Final Improvement on Baseline Performance: Utilize MTCNN Face detection to extract faces from the image, generating embeddings using FaceNet and finally using Milvus to search for the target images with a count of 10. 

#####1.	CREATING BASELINE PERFORMANCE:
Before we dive deeper into finding the embeddings on an image, let’s understand what embeddings are and how it helps us. When we are dealing with images, the machine interprets an image in the form N x N matrix with each cell containing the pixel value of the image. 
 
To find the similarity between two images, one of the ways to go about it is by using a distance measure. Although this method works pretty well to find similarities in the images, it fails to account for the different variations such as size, rotation, etc. of the same image. To account for this, we have various classical descriptors (SIFT, SURF) that help us to extract the salient features of a given image. This method has been widely used in the past decade for hand-crafted content-based image retrieval. However, with the recent developments in deep neural networks, especially convolutional neural networks, we can achieve the state-of-the-art results in classification problems. These same CNN layers can be utilized to generate a feature vector that extracts features from an image, invariant to its geometrical transformation and instance. These feature vectors contain the key points of the image from our embedding. Embeddings are vectors that can be painted on the cartesian plane representing an image. Hence, in our project, we are going to utilize ResNet50 to generate these key point feature vectors that will be later utilized by our machine learning model to generate similar images. 

ResNet50 (Residual Network): ResNet50 was an improvement over the AlexNet which helped in training an extremely deep neural network by solving the problem of vanishing gradient. The deep neural network here means that we can achieve great results even if our model has hundreds of fully connected layers in the network. But we will not dive deeper into this as we are concerned with the feature vector that is generated by the convolutional layers of this model. In short, the ResNet50 model is used to find the embeddings of an image in our project. Let’s see a bit more into the ResNet50 Architecture to understand what is happening behind the scenes and how we are going to get the embeddings. 
 
<p align="center">
  <img src="(https://user-images.githubusercontent.com/34485564/165446770-818da860-bd0b-4189-9d2b-c2d5dba203bc.png)" />
</p>

Figure 1 ResNet- 50 Convolutional Layer

The model consists of 5 stages with each convolutional block consisting of three convolutional layers (convolution + Batch Normalization + ReLU) and each identity block containing 3 convolutional layers (Convolution + Batch Normalization). The identity block here is what was introduced by the authors of ResNet50. This block helps us to perform ‘skip connections. Skip Connection here refers to adding the original input to the output of the convolutional block. This is what helps ResNet50 help to deal with the problem of vanishing/exploding gradient problems. 
In our project, we obtain the embeddings/feature vector of size 2048 at the end of stage 5 by adding an Average Pooling layer. 

<p align="center">
  <img src="(https://user-images.githubusercontent.com/34485564/165446804-e6cc8df1-d975-4678-b86f-aea49b9e9043.png)" />
</p>
 
Figure 2 Average Pooling Layer Added at the end of Stage 5 and Parameter Description

Once we obtained this feature vector, we now pass it through our machine learning model to train it. The model that we used for baseline performance is K- Nearest Neighbors. 
K- Nearest Neighbors: This algorithm is an easy-to-implement supervised machine learning algorithm that will help us to find images that most closely resemble our given search query based on the distance. We utilize this algorithm from the sklearn library. 
 
######Results:
The result here shows the similarity search done by our model for Albert Costa, Spanish’s former tennis professional. 
 
As we can see our model couldn’t classify properly and we could only find 1 out of the 6 Albert Costa images. Hence, we move forward to our initial improvement. 
2.	INITIAL IMPROVEMENT ON BASELINE PERFORMANCE:
When we check the LFW Dataset we see that the images in this dataset have a lot of noise in their background, in the form of unknown faces, objects, etc. So, the embeddings that we found earlier take into account the noise associated with a given image. To improve our image search query, one of the ways we adopted is using MTCNN to help us extract the relevant information in the form of face extraction and provide this through our base model.
 
Figure 3: Architecture Design for Initial Improvement

MTCNN (Multi Tasked Cascaded Convolutional Neural Network):  This is a face detection model that was based on the paper by Kaipeng Zhang. The paper describes a deep cascaded multi-task framework that takes features from different sub-models, namely P- Net, R-Net, O-Net), to boost their correlating strengths. This model works by generating a pyramid of images for a single (to account for different sizes of a face in an image in the form of multiple scaled copies). These pyramids of images are then fed to the P-Net which then gives us the bounding boxes over the faces. To refine this, the model removes the bounding boxes of low confidence level and performs non-maximum to give us the final output. This output is then again fed to R- Net and O-net with the same aforementioned procedures. The output of this model is an image with a bounding box over the face. 
We use these bounding boxes to crop our LFW Dataset. The K-NN model is then again trained on the embeddings based on the faces of the LFW Dataset. To search a query, we pass the input image through face extraction and embedding layer as shown in figure3. 

#####Results:
We can see our model was able to find more pictures of Albert Costa, thus improving on our previous performance. 

<p align="center">
  <img src="(https://user-images.githubusercontent.com/34485564/165446858-9b327446-80d7-4187-94a5-cf4d59dedf80.png)" />
</p>

This is still not a good performance as we could only find 3 out of 6 Albert Costa Images in the LFW Dataset. Hence, we move forward to our second and final improvement.


#####3.	FINAL IMPROVEMENT ON THE BASELINE PERFORMANCE:
The results from the previous method didn’t give us satisfactory accuracy. Therefore, for our final improvisation, we utilize MTCNN, FaceNet, and Milvus. Now in the previous improvement, we utilized MTCNN to extract faces from the image and find the embedding based on these new images. However, in this improvisation, we are going to pass these extracted face images to FaceNet.
FaceNet: This is a deep neural network that is used to extract salient features from an image of a face. FaceNet has adopted two types of CNN in its architecture namely, Zeiler & Fergus architecture and GoogLeNet Style Inception Model.  It works with the help of a special loss function, called the triplet loss. To measure this loss, we use three images, namely anchor, positive and negative. The purpose of the triplet loss is to:-
1.	minimize the distance (L-2 Distance) between the anchor (target image) and negative image
2.	maximize the distance (L-2 Distance) between the anchor (target image) and positive image 
The above intuition to finding embeddings in such a way that reduced the triplet loss. The final embedding is of size 256 which is formed by flattening the feature vector using Average Pooling Layer, in our case. 

Another change that we will have from our previous approach is that we will use Milvus instead of KNN. Milvus is an open-source vector database that is used to build a reverse image search system, for our project. Our embeddings for the LFW Dataset are sent to the Milvus Database (cluster). For the query search, we pass the image through MTCNN and FaceNet again to fetch the embedding layer. This layer is then sent to Milvus which is compared against the data in the cluster to find a similar image. 
 
Figure 4 Architecture Design for Final Improvement

#####Results:
We see that all the images of Albert Costa have been found by our model. Thus, we can see the improvement over our previous approach towards Reverse Image Search. 
 
<p align="center">
  <img src="(https://user-images.githubusercontent.com/34485564/165446887-407f0a16-e3a7-4714-be9a-43f94178855d.png)" />
</p>

REFERENCES:
[1] https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
[2]https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff
[3] https://milvus.io/docs/image_similarity_search.md






### Group Members
1. Eashan Kaushik
 - NYU Email: ek3575@nyu.edu
 - N#: N19320245
 - GitHub Email: eashank16@gmail.com
2. Rishab Redhu
 - NYU Email: rs7623@nyu.edu
 - N#: N18032325
 - GitHub Email: rs7623@nyu.edu
3. Rohan Jesalkumar Patel
 - NYU Email: rp3617@nyu.edu
 - N#: N13328358
 - GitHub Email: rp3617@nyu.edu
 
### AWS Contact
Email: ek3575@nyu.edu

