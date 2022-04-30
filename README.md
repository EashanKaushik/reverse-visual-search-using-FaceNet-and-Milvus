# Reverse-Visual-Search

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/165983370-99b91cdf-d3c4-40d0-98a0-ffbd5a2c3782.png" />
  <br>
  Figure 1: Reverse Visual Search
</p>

## PROBLEM STATEMENT

We have all played the game of “spot the difference” in which we need to find differences between two similar images. To build upon the context, can you find images that are similar to a given image? The google reverse image search is an apt description of what we are trying to building in this project. Our problem statement is to find N similar images, given an input image.

## DATASET

In this project, we are going to utilize the LFW Dataset (Labelled Faces in the Wild), a compilation of face pairs of positively and negatively weighted images samples collected by the researchers of the University of Massachusetts. The dataset comprises 13233 images of 5749 people with 1680 people having two or more images. LFW Dataset is the benchmark for pair matching, also known as face verification. For our project, we are going to build our model on this dataset and later utilize this model for a reverse visual search on a given face image.

<p align="center">
  <img src="https://user-images.githubusercontent.com/34485564/165446646-7d839b67-fe85-4cbc-84ab-fd88576510ea.png" />
  <br>
  Figure 2: LFW Dataset
</p>

## ARCHITECTURE

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089055-a00f51dd-5f28-4886-83fc-81caf7d99554.png" />
  <br>
  Figure 3: Model Architecture
</p>

Steps for reverse visual search: 
1.	Generating the embeddings for entire dataset.
2.	Storing these embeddings in a vector database.
3.	Generating the embedding for query image. 
4.	Searching the 20 closest neighbors in the vector database.
5.	Giving the results.

For the purpose of developing a model from ground up we have developed three different architectures and tried to get better results in each iteration. Different models are discussed below:

1.	Developing a Baseline Model: In this model we have used ResNet50 for generating the embeddings of an image and providing it as an input to searching ML model, in this case K-Nearest Neighbor.
2.	Initial Improvement on Baseline Performance: Utilizing MTCNN Face detection to extract target images (faces) and utilizing ResNet50 along with K- Nearest Neighbor (similar to the baseline model) for visual search.
3.	Final Improvement on Baseline Performance: Utilize MTCNN Face detection to extract target faces from the image, generating embeddings using FaceNet and finally using Milvus to search for the similar images. 

## BASELINE MODEL

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089097-ae53a0f3-7e7b-4494-8f6b-74c7ddd93361.png" />
  <br>
  Figure 4: Baseline Model
</p>

### Results for Baseline Model: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089130-b7e1c21d-52ef-4e9f-be50-8f2fc7bedde6.png" />
  <br>
  Figure 5: Carmen_Electra
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089134-4db97470-4d21-406b-8983-53094e6d26b5.png" />
  <br>
  Figure 6: 20 Similar Faces for Carmen Electra
</p>

## IMPROVED MODEL

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089187-589ef35f-139d-4837-87db-a770ee476b7f.png" />
  <br>
  Figure 7: Improved Model
</p>

### Results for IMPROVED Model: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089130-b7e1c21d-52ef-4e9f-be50-8f2fc7bedde6.png" />
  <br>
  Figure 8: Carmen Electra
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089196-e05de61a-4251-4def-80ff-41f92ab5017f.png" />
  <br>
  Figure 9: 20 Similar Faces for Carmen Electra
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089254-0162d2e4-b7cf-4189-bb0c-dcf13224c510.jpg" />
  <br>
  Figure 10: Albert Costa
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089271-fc799085-ef01-42a3-b335-8a89516bbf1f.jpg" />
  <br>
  Figure 11: 20 Similar Faces for Albert Costa
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089284-8697a6ad-0ab3-49a0-a413-ed47be762962.jpg" />
  <br>
  Figure 12: Angela Bassett
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089310-0b0d95c1-974b-4a1f-941b-2adc91f07ff0.jpg" />
  <br>
  Figure 13: 20 Similar Faces for Angela Bassett
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089327-063be04d-30a2-433b-a52c-241bbe0cd462.jpg" />
  <br>
  Figure 14: Arminio Fraga
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089342-d74d3ce8-e62c-4c72-bd5c-dc109ff66ba1.jpg" />
  <br>
  Figure 15: 20 Similar Faces for Arminio Fraga
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089372-9eb3688b-3a40-4ee0-af35-4a7674e2abe9.jpg" />
  <br>
  Figure 16: Billy Crystal
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089385-11d4100e-1111-4b95-ba2c-47756c5c7dd8.jpg" />
  <br>
  Figure 17: 20 Similar Faces for Billy Crystal
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089394-85fff03f-e537-4d49-ae06-4ab0c5132fa8.jpg" />
  <br>
  Figure 18: Bob Graham
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089410-af1371e6-f1ee-42bf-8898-07bd90003c73.jpg" />
  <br>
  Figure 19: 20 Similar Faces for Bob Graham
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089423-b6502a84-5d53-4fe4-ad89-17f3b6f7ee87.jpg" />
  <br>
  Figure 20: Boris Becker
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089440-fdcb68f8-a96b-4827-b3a2-54236dfe194c.jpg" />
  <br>
  Figure 21: 20 Similar Faces for Boris Becker
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089463-2a1ef6e2-688b-4525-ae60-66280e503dfd.jpg" />
  <br>
  Figure 22: Bulent Ecevit
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089471-7bd993f1-5dcb-41d1-9958-4c0e0bdb6ff1.jpg" />
  <br>
  Figure 23: 20 Similar Faces for Bulent Ecevit
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089483-6cd91648-fa60-4c45-ae0f-f8f4f1994f4d.jpg" />
  <br>
  Figure 24: Calista Flockhart
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089493-a197b1fd-4723-4978-b331-50d90732eed3.jpg" />
  <br>
  Figure 25: 20 Similar Faces for Calista Flockhart
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089506-130aed25-8139-4c0a-86d0-4a49672715fb.jpg" />
  <br>
  Figure 26: Cameron Diaz
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/166089514-56b92b1c-4805-4afb-ba54-13997d973470.jpg" />
  <br>
  Figure 27: 20 Similar Faces for Cameron Diaz
</p>

# Steps to Replicate the Experiment:

This project was completed on multiple machines: 

1. Colab Pro (High Ram and GPU) for running notebooks for training.  

a. model-training-baseline.ipynb  
b. model-training-facenet.ipynb  

2. EC2 Instance for Milvus.

a. model-training-milvus.ipynb  
b. query/Improvement-final-Milvus.ipynb  

3. Local Machine for running notebooks for generating outputs.

a. query/Preprocessing-Queries.ipynb  
b. query/Baseline_Model-Output.ipynb  
c. query/Improvement-initial-Facenet&Knn.ipynb  
d. query/Improvement-final-FaceNet.ipynb  

### Step 1

Download the folder with all generated output for embeddings and query images: https://drive.google.com/drive/folders/1u4ESZFtd0NeOBaIDdRS3Bc2bgKr2qoEU?usp=sharing, all these files must be in a folder named "reverse-visual-search".

### Step 2

Download the LFW dataset from here: 

### Note  
There are two phases to the project   
    (1) Training - This part was done on Colab Notebook and a bit difficult to replicate as it takes time to generate embeddings. The ipynb files are placed in the training folder of the drive link in step 1.   
    (2) Output Queries - This part was done on local machine and is quick and easy to replicate this part is placed in query folder of the drive link given in step 1. 

### Step 3

Please note you can skip this step as it takes time to generate embeddings, you can skip to step 4 as all the outputs from this step are provided in Step 1 drive link. However if you want to test this part please run on Colab Pro having High Ram and GPU run time. To run these files you need to run the ipynb in proper sequence or else you might get an error, these ipynb files are present in training folder in drive link given in step 1.

    (1) model-training-baseline.ipynb (Colab)  
    (2) model-training-facenet.ipynb (Colab)  
    (3) model-training-milvus.ipynb (EC2)  

### Step 4

In this step we will get 20 similar faces for 10 queries given in inputs folder in the drive link given in step 1. The ipynb notebooks are present in query folder in drive link. Sequence to get desired output.

    (1)

## REFERENCES:

[1] https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33  
[2] https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff  
[3] https://milvus.io/docs/image_similarity_search.md  
[4] https://aws.amazon.com/blogs/machine-learning/building-a-visual-search-application-with-amazon-sagemaker-and-amazon-es/  
[5] http://vis-www.cs.umass.edu/lfw/  


## Group Members

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
 
## AWS Contact

Email: ek3575@nyu.edu

