# Paper: [Unsupervised Feature Learning via Non-Parametric Instance Discrimination]()

## Outline

## Introduction
In this study, the researchers explore the idea that neural networks, even when trained without explicit class labels, can still capture visual similarities between data instances. This concept goes beyond conventional supervised learning, where networks are directed to differentiate between predefined classes. Instead, the researchers aim to learn a feature representation that captures similarities between individual instances, regardless of their class memberships.

The approach they take involves formulating the problem as a non-parametric classification task at the instance level. This means that instead of categorizing data into classes, they aim to distinguish between individual data instances. To handle the computational challenges arising from a large number of instance "classes," they employ a technique called noise-contrastive estimation. This technique helps in effectively training the model despite the vast number of instances.

The experimental outcomes of their approach are quite remarkable. Under unsupervised learning conditions, their method outperforms the current state-of-the-art in ImageNet classification by a significant margin. Notably, their method continues to exhibit improved performance as more training data is used and as better network architectures are employed. 

Additionally, by fine-tuning the features learned through this non-parametric approach, the researchers achieve competitive results in scenarios where there is some labeled data available (semi-supervised learning) and even in object detection tasks. 

What's particularly noteworthy is that their non-parametric model is compact in terms of storage requirements. For instance, with only 128 features per image, the method requires just 600MB of storage to handle a million images. This efficiency enables efficient nearest neighbor retrieval during runtime, which can be advantageous in real-time applications.

In essence, the researchers have demonstrated the efficacy of learning feature representations that capture similarities between individual instances rather than predefined classes. This approach leads to impressive performance gains in unsupervised learning tasks, and the learned features can be further adapted to perform well in scenarios involving some labeled data or specific tasks like object detection. The added benefit of storage efficiency makes their approach quite practical and promising for real-world applications.

The focus is on exploring whether it's possible to learn a meaningful metric of similarity among individual instances, rather than semantic categories, through discriminative learning. The idea is to treat each image as unique and distinct, even within the same semantic category. By training a model to discriminate between individual instances without considering class labels, it's hypothesized that the resulting representation could capture the apparent similarity between instances, much like how class-wise supervised learning captures similarity between classes.

This approach is appealing because it allows leveraging advances in discriminative supervised learning, potentially benefiting from new network architectures and techniques. However, a significant challenge arises due to the large number of "classes" in this scenario. Instead of dealing with a fixed number of semantic categories, now each image becomes a potential "class," leading to an extremely large number of classes, such as 1.2 million in the case of ImageNet.

To address this challenge, the researchers employ a technique called noise-contrastive estimation (NCE) to approximate the full softmax distribution, which would be unfeasible in such a high-dimensional scenario. NCE is a method used to simplify the computation of the softmax function when dealing with a large number of classes. Additionally, a proximal regularization method is used to stabilize the learning process, enabling effective training despite the large number of "classes."

To assess the effectiveness of unsupervised learning, prior research often employed linear classifiers like Support Vector Machines (SVMs) to link the learned features to categories for classification during testing. However, it's not entirely clear why features learned for a specific training task would inherently be linearly separable for an unrelated testing task.

This study introduces a different approach, favoring a non-parametric strategy for both training and testing. The core idea is to treat instance-level discrimination as a metric learning problem. Instead of relying on learned weights in a network, the features of each instance are directly stored in a discrete memory bank. This creates a metric space where distances (similarities) between instances are calculated straightforwardly. During testing, classification is conducted using k-nearest neighbors (kNN) based on the learned metric. This approach ensures consistency between training and testing, as both are centered on the same metric space of image relationships.

###### non-parametric
The term "non-parametric instance discrimination" refers to the approach used in the research where instances (individual data points, in this case, images) are treated in a non-parametric manner to achieve discrimination.
1. Non-Parametric: In statistics and machine learning, a non-parametric method is one that doesn't make strong assumptions about the underlying distribution of the data. In this context, "non-parametric" indicates that the approach doesn't rely on predefined model parameters. In contrast, "parametric" methods have a predetermined structure with a fixed number of parameters that are learned from the data.

Let's say we're using linear regression to predict house prices based on the number of bedrooms. In a parametric approach, we assume that the relationship between the number of bedrooms (input) and the house price (output) can be represented by a linear equation: price = w * bedrooms + b, where w and b are parameters to be learned during training.

In a non-parametric approach, we might not assume any specific functional form for the relationship. Instead, we might use methods like k-nearest neighbors or kernel density estimation, where predictions are based on the properties of neighboring data points rather than a fixed equation with parameters. 

2. Instance Discrimination: Discrimination in this context means distinguishing or differentiating between different instances (data points). The goal is to learn features that can separate instances from each other based on their intrinsic characteristics.

In other words, the method aims to enable instances to be distinguished from each other based solely on their inherent similarities and differences, without making specific assumptions about the distribution of the data or employing fixed model parameters. The method stores instance features directly in a memory bank rather than relying on predefined weights in a neural network. This allows the approach to learn a flexible notion of similarity without a fixed model structure.

The similarity between instances is calculated based on the learned features (which involves adjusting model parameters), but it doesn't involve adjusting any additional parameters. Instead of using model parameters to calculate similarity, the method stores the features of each instance in a memory bank. When calculating similarity between instances, it directly compares the stored feature representations. This is what makes it "non-parametric" in this context – the similarity calculation is not dependent on adjusting model parameters.

###### Top-1 and top-5 classification error 
Metrics commonly used to evaluate the performance of image classification models, especially in large-scale datasets like ImageNet. These metrics measure the accuracy of a model's predictions by considering the correct class among the top predicted classes.

1. Top-1 Classification Error: This metric calculates the percentage of images for which the true class label is not among the model's top predicted class. In other words, it measures the proportion of cases where the model's most confident prediction is incorrect.
2. Top-5 Classification Error: This metric is more lenient than top-1 error. It calculates the percentage of images for which the true class label is not among the top 5 predicted classes. It takes into account the possibility that the true label might be ranked a bit lower in the predictions, but still within the top 5.


Suppose we have an image classification model trained to predict the class of objects in images. The model assigns a probability score to each class label for a given image. Suppose we have an image of a cat, and the model predicts the following probabilities for the top 5 classes:
1. Cat: 0.75
2. Dog: 0.12
3. Rabbit: 0.05
4. Bird: 0.04
5. Elephant: 0.02
Top-1 classification error: The true label is "Cat" for the image, but the model's top predicted class is also "Cat" (0.75 probability). So, the top-1 classification error is 0%, as the true label is among the top predicted classes.
Top-5 classification error: The true label is "Cat," and it is among the top 5 predicted classes. So, the top-5 classification error is also 0%.

Now, let's consider an example where the model's predictions are not accurate:
1. Dog: 0.35
2. Cat: 0.25
3. Elephant: 0.18
4. Rabbit: 0.15
5. Bird: 0.07
Top-1 classification error: The true label is "Elephant," but the model's top predicted class is "Dog" with a probability of 0.35. So, the top-1 classification error is 100%, as the true label is not among the top predicted classes.
Top-5 classification error: The true label is "Elephant," and it is not among the top 5 predicted classes. So, the top-5 classification error is 100%.

Lower values for these metrics indicate better performance, as we want the true label to be highly ranked among the model's predictions.

###### Generative Models
The aim of generative models is to understand and capture the underlying data distribution in order to generate new, similar data points. Traditional generative models like Restricted Boltzmann Machines (RBMs) and Autoencoders have been used for this purpose. These models produce latent features that can also be beneficial for tasks like object recognition. More recent techniques, such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), not only enhance the quality of generated data but also improve the learned features.

###### Self-supervised Learning
This focuses on utilizing inherent structures within the data to formulate predictive tasks that can train a model. The idea is to create tasks where the model has to predict certain aspects or components of an instance based on the other parts. For images, these tasks could involve predicting context, counting objects, recovering missing parts of an image, etc. Similarly, for videos, tasks could include predicting future frames or maintaining the consistency of egomotion (motion of the viewer). Combining multiple self-supervised tasks has also been explored to achieve better visual representations.

Even though self-supervised learning can capture relationships among different parts or aspects of an instance, it's not always clear why a specific self-supervised task would directly benefit semantic recognition (recognizing objects or meanings) or which task would be most effective for this purpose. In other words, while self-supervised learning might help the model understand various aspects of the data, it's not straightforward to determine how those aspects would contribute to better semantic understanding or recognition of objects or meanings.

###### Metric learning
In the context of this paper, "Metric Learning" refers to the concept of defining a measure of distance or similarity between data instances based on their feature representations. For a given feature representation function F, the distance between two instances x and y is calculated using the Euclidean distance (∥F(x) − F(y)∥).

Metric learning is a well-studied area in machine learning and has been applied in various tasks like face recognition and person re-identification. In such tasks, the goal is often to learn a metric that can distinguish between different instances (like different faces or people) even when the classes at test time are different from those at training time. Metric learning methods typically require some form of supervision to learn this distance metric.

However, the approach proposed in this paper takes a different path. It leverages the idea of unsupervised learning, where no human-provided labels are used during training. The paper's approach learns features and the associated metric in an unsupervised manner, without any form of human annotations. This metric learning process is based solely on the inherent structure and patterns in the data, allowing it to capture the similarity among instances without any explicit class labels.

###### Exemplar CNN
The Exemplar CNN and the approach proposed in this paper share some similarities, but they have a fundamental difference in their underlying methodology.

The Exemplar CNN also aims to learn feature representations that capture the similarity among instances. However, it follows a parametric approach during both training and testing. This means that the Exemplar CNN relies on predefined model parameters, which are adjusted through training to learn the desired features. These learned features are then used to calculate the similarity between instances during testing.

In contrast, the method presented in this paper is non-parametric in nature. This makes the approach more flexible and allows it to work without relying on predefined model weights.

The paper conducts experiments to empirically demonstrate the difference between the parametric Exemplar CNN and their non-parametric approach. The Exemplar CNN is also noted to be computationally demanding for large-scale datasets like ImageNet, while the proposed non-parametric approach offers advantages in terms of efficiency and memory usage.

Exemplar CNN focuses on learning features that can distinguish between individual instances rather than predefined classes
1. Training phase: During the training phase, the Exemplar CNN is trained on a dataset of images.Unlike traditional classification tasks where images are labeled with specific classes, in the Exemplar CNN, the images are treated as individual instances without class labels. The CNN is trained to map each input image to a feature representation in a higher-dimensional space. If it has 100 images, then the classes will be hundred. Per class one image would not work for CNNs, heavy augmentation is needed for every image.
2. Feature Learning: The CNN learns to encode distinctive features from the input images. These features aim to capture the intrinsic characteristics of each instance, allowing the network to differentiate between different images
3. Similarity Calculation: After training, the learned feature representations are used to calculate the similarity between pairs of instances. The similarity is typically calculated using some distance metric, such as Euclidean distance, between the feature vectors of two instances. Instances that have similar features will have a lower distance and thus a higher similarity score.
4. Testing Phase: In the testing phase, the Exemplar CNN uses the calculated similarity scores to make predictions about the relationships between instances. For example, it can identify which instances are more similar to each other based on the calculated distances.
5. Limitations and Differences from Non-Parametric Approach: The Exemplar CNN adopts a parametric approach, meaning that it learns and relies on model parameters during both training and testing. It requires training the CNN on a large amount of data, which can be computationally demanding and memory-intensive, especially for large-scale datasets like ImageNet. The learned parameters in the Exemplar CNN contribute to the feature extraction and similarity calculation.

Compared to the non-parametric approach discussed in the paper, which directly calculates the similarity from the stored feature representations without relying on predefined model parameters, the Exemplar CNN uses a traditional CNN architecture and parametric learning framework. This distinction has implications for efficiency, scalability, and flexibility, as outlined in the paper's comparison.

## Approach in general
1. Embedding Learning: The primary objective is to learn an embedding function, denoted as v = fθ(x), where x represents an image and v represents the learned feature representation. The embedding function is implemented through a deep neural network with parameters θ. The aim of this embedding is to create a metric space within the image data, where the distance between two instances x and y in the feature space is defined as dθ(x, y) = ∥fθ(x) − fθ(y)∥. This metric helps measure the similarity or dissimilarity between instances.

2. Instance-Level Discrimination: A classifier is trained to distinguish between the individual instance classes. In this context, the goal is to have the network learn to differentiate between instances based solely on their intrinsic characteristics without any predefined class labels.

3. Objective of the Approach: The main objective is to learn feature representations that capture visual similarity between instances in an unsupervised manner. The embeddings generated through the neural network should ideally position visually similar images closer to each other in the feature space, facilitating easy discrimination between instances.

Imagine we have a dataset of images, and we want to learn a feature representation for these images without any labeled class information. The goal is to make similar images have feature representations that are close in the learned feature space.
1. Feature Learning: We have a deep neural network (CNN) with parameters θ. This network takes an image x as input and produces a feature representation v = fθ(x). This feature representation should capture the essence of the image's content in a way that similar images have similar feature vectors.
2. Instance-Level Discrimination: Instead of traditional supervised learning where we have labeled classes, in this approach, we treat each image as its own class. So, we have as many classes as there are images in our dataset.
3. Non-Parametric Classification: The idea is to classify instances (images) based on their feature vectors. Instead of using traditional softmax with class-specific weight vectors, we replace the weight vectors with the feature vectors themselves. This way, we directly compare feature vectors to determine similarity. For instance, let's say we have three images: Image A, Image B, and Image C. We want to see if Image A is similar to Image B or Image C. We compute the dot product between the feature vectors: vA.T * vB and vA.T * vC. If the dot product between vA and vB is higher than vA and vC, then you conclude that Image A is more similar to Image B.
4. Temperature Parameter: To control the sensitivity of the comparison, we introduce a temperature parameter τ. A higher value of τ makes the probabilities more uniform, while a lower value makes them sharper. This parameter influences how we determine the similarity based on dot products.
5. Negative Log-Likelihood Loss: We want to adjust the network's parameters θ in a way that makes it good at classifying instances based on their feature vectors. To do this, we calculate the negative log-likelihood loss. This loss quantifies how well the network's predictions match the true similarity between instances. For each image, we calculate the probability that it's most similar to itself (the positive class) and the probabilities for other images (negative classes). The loss is the negative log of the predicted probabilities.
6. Memory Bank: To avoid computing feature vectors repeatedly, we maintain a memory bank V where we store the computed feature vectors for each image. This speeds up the process since we don't need to recompute feature vectors during training.

## NPID
1. Network Architecture: We have a CNN model (let's call it the embedding model) that takes an image as input and produces a feature vector.
2. Memory Bank: We maintain a memory bank where each entry corresponds to a stored feature vector. Initially, these vectors can be randomly initialized or set to some default values.
3. Loss Calculation: In each training iteration, we process a batch of images through the embedding model, obtaining their feature vectors. For each image in the batch, we compare its feature vector to all the feature vectors stored in the memory bank. This involves computing the pairwise distances between the feature vectors. We then apply a similarity metric, such as the dot product or cosine similarity, to calculate the similarity between the feature vector of the current image and all the stored feature vectors. For each image, the most similar stored feature vector (with the highest similarity score) becomes the positive example, and all other stored feature vectors become negative examples. The loss function encourages the positive similarity score to be high and the negative similarity scores to be low. One common loss function used for this purpose is the InfoNCE (Normalized Cross-Entropy) loss.
4. Memory Bank Update: After computing the loss and updating the model's parameters, we update the memory bank using a specific strategy.
5. Ground Truth for Loss: The ground truth for calculating the loss comes from the positive and negative examples determined during the loss calculation step. For each image, we have one positive example (the most similar stored feature vector) and multiple negative examples (other stored feature vectors). The loss function is computed using the similarity scores between the positive and negative examples, encouraging higher similarity between positive examples and lower similarity between negative examples.

In summary, the ground truth for calculating the loss in non-parametric instance discrimination is based on the similarity scores between the current batch's feature vectors and the stored feature vectors in the memory bank. The loss function aims to maximize the similarity between positive examples and minimize the similarity between negative examples, facilitating the learning of a feature space where visually similar images have higher similarities.

## Learning with a memory bank:
1. Memory Bank: The memory bank V is a collection of feature vectors of all the images that have been processed so far. Each vj represents the feature of the j-th image in the dataset.
2. Learning Iteration: In each learning iteration (typically corresponds to a training batch), the following steps are performed:
   a. Feature Extraction: The feature representation fi (output of the embedding network for the i-th image xi) is computed.
   b. Optimization: The network parameters θ are updated through stochastic gradient descent (SGD) using a loss function that encourages the model to produce meaningful feature representations. This loss function could be the InfoNCE loss as described earlier.
   c. Memory Bank Update: The updated feature representation fi is then stored in the memory bank V. This is done by replacing the existing vector vj in the memory bank corresponding to image xi with the updated feature fi. This update is denoted as fi → vi, meaning the i-th entry of the memory bank is replaced with the updated feature fi.
3. Initialization of Memory Ban*: The memory bank V is initialized at the beginning with random vectors. This is often done to kickstart the process and make sure that the memory bank has a reasonable starting point.

The vector in the memory bank corresponding to an image is updated within each learning iteration (often within each batch). This means that the memory bank gets updated multiple times within an epoch. The specific update strategy (replacing the vector with the new feature) is applied directly for each image.

## Example
1. Say, we have 10 samples in a dataset X = {bird1, bird2, cat1, cat2, dog1, dog2, rat1, rat2, horse1, horse2}
2. We have a CNN embedding model f with some parameters.
3. We have a memory bank V with size 10 same as the real dataset and initialized randomly.
4. Say, in a batch we have 5 samples {bird1, cat1, dog1, rat1, horse1}
5. We pass this batch though f and we will get {v1, v2, v3, v4, v5} i.e feature representations of the samples in the current batch
6. Let's talk about datapoint-1: we will take the dot product of v1 with every feature vector in the memory bank. So, the non-parametric classifier is: the probability of v1 being recognized as i-th example is:
![equation of non-parametric softmax]()
7. Then the learning objective is to minimize the negative loglikelihood over the training set:
![equation of log-likelihood]()
8. Computing the non-parametric softmax is cost prohibitive when the number of classes n is very large, e.g. at the scale of millions. Popular techniques to reduce computation include hierarchical softmax, noise-contrastive estimation (NCE), and negative sampling. Here authors have used NCE. The basic idea is to cast the multi-class classification problem into a set of binary classification problems, where the binary classification task is to discrimi- nate between data samples and noise samples. We have 5 samples in the current batch, the negative/noise samples will be selected from remaining dataset. For this example it will be from {bird2, cat2, dog2, rat2, horse2}. So, the probability that feature representation v in the memory bank corresponds to the i-th example under this model is:
![equation NCE]()
10. Authors formalize the noise distribution as a uniform distribution is Pn = 1/n, here it will be 1/10. This means that every sample in the dataset has 0.1 probability to get selected as noise sample.  Authors assume that noise samples are m times more frequent than data samples.
11. Once the NCE values have been calculated for each instance (both real data instances and noise samples), they are used to define the training objective. The goal is to minimize the negative log-likelihood of the data samples being classified correctly as real data instances and the noise samples being classified correctly as noise.
12. Here, the NCE values are used to define the NCE loss function, which is formulated as:
JNCE(θ) = −EPd [logh(i,v)] − m · EPn [log(1 − h(i, v'))]
13. JNCE(θ) is the NCE loss function that the model aims to minimize.
14. Pd represents the actual data distribution, and Pn represents the noise distribution.
15. h(i, v) is the posterior probability that the instance i with feature v is from the data distribution. the posterior probability of sample i with feature v being from the data distribution (denoted by D = 1) is
![equation of h(i,v)]()
16. v' is the feature from another image randomly sampled according to the noise distribution Pn.
17. The first term in the loss accounts for correctly classifying data instances, and the second term accounts for correctly classifying noise instances. The goal of optimization is to adjust the model's parameters (θ) to minimize this NCE loss.
18. So we will get the dot prodcuts for v1 with every in memory bank, p(first element in memory bank | v1) = their dot product / Z, p(second element in memory bank | v1) = their dot product / Z,...., and on this we will minimize negative log likelihood.
19. By optimizing this NCE loss function, the model learns to distinguish between real data instances and noise samples. The learned feature representations are expected to map similar instances closer together and dissimilar instances farther apart in the feature space, leading to improved discriminative power and feature quality.
20. However, directly computing Zi for all instances in the training set can be computationally expensive (O(n) complexity).
21. To address this computational challenge, the paper proposes an approximation using Monte Carlo estimation. They treat Zi as a constant and estimate its value using a random subset of indices jk, which reduces the complexity to O(1) per sample.
![equation of monte carlo approximation]()
22. Here for this example in each iteration/epoch, we would calculate the value of Zi once for that specific iteration. Consider 20 epochs, we would calculate Zi 20 times throughout the training process. The idea is to calculate it once for each epoch, as the value of Zi doesn't change within an epoch and remains constant for that epoch's computations. This helps in reducing the computational complexity and still achieves competitive performance.
23. We should ensure that the negative samples or noise samples used for the computation of Zi are not drawn solely from the same batch. This is to ensure that the noise samples are representative of the entire dataset and that the NCE loss is calculated accurately. If all the negative samples come from a single batch, it might lead to biased or incorrect estimations of Zi, which could impact the learning process and the quality of the learned features. To maintain a fair and accurate estimation of Zi the negative samples should be diverse and cover examples from various batches and classes.
24. It's generally acceptable if a small portion of the negative samples used in the NCE loss computation come from the current batch. Including a few negative samples from the same batch can help provide a more localized context for the model to discriminate between similar instances within the batch. However, the majority of negative samples should come from outside the batch to ensure diversity and accurate estimation of Zi
25. This way NCE reduces the computational complexity from O(n) to O(1) per sample.


## Proximal Regularization
A technique employed to enhance the training process of the neural network in the context of instance discrimination. It's used to address certain challenges that arise due to the nature of the dataset and the training setup.
1. In instance discrimination, there is typically only one instance per class in the dataset. Also, during each training epoch, each class is visited only once. This can lead to significant fluctuations in the training process due to the randomness introduced by the limited number of instances.
2. To stabilize the training process and encourage smoother dynamics during learning, the paper employs a technique called proximal optimization. This technique introduces an additional term to the loss function that encourages the new feature representations to be close to the feature representations from the previous iteration.
3. Given the feature representations of data instances computed by the network at iteration t, and the  feature representations from the previous iteration t-1, the loss function for a positive sample from the real data distribution
![loss function for positive sample]()
4. h() - represents the posterior probability that instance with it's feature vector is from the data distribution.
5. Lambda is a hyperparameter that controls the strength of the regularization. It determines how much the new feature representations (current iteration) should stay close to the previous ones (previous iteration)
6. And the last is squared L2 distance between the feature representations at the current and previous iterations.
7. With the inclusion of proximal regularization, the ultimate objective function for training becomes:
![loss function regularization]()
8. This objective function combines the NCE loss term, which focuses on distinguishing real data instances from noise samples, and the regularization term that encourages smoothness and stability in the feature representations across iterations.
9. By using proximal regularization, the learning process becomes less sensitive to the randomness of instance selection and aims to achieve more stable and consistent learning dynamics. This can help improve the quality of learned feature representations.

## Weighted k-Nearest Neighbor Classifier
The weighted k-Nearest Neighbor (kNN) Classifier is a method used for classifying test images based on their similarity to the training images. In this context, it's used to classify images using the learned feature representations obtained from the neural network.

Suppose we have a dataset of animal images with four classes: Birds, Cats, Dogs, and Horses. We've trained a neural network to extract feature representations from these images. We also have a memory bank V with feature vectors of the training images.

1. Feature Extraction: For a test image, the trained neural network model is used to extract a feature representation by passing through the network.
2. Cosine Similarity: The cosine similarity is calculated between the extracted feature and the feature of each image in the memory bank. Cosine similarity is a measure of how similar the directions of two vectors are.
3. Top-k Nearest Neighbors: The k images with the highest cosine similarity values to the test image are selected as the top k nearest neighbors Nk. We select the 3 images with the highest cosine similarity values to test feature vector. These images are our nearest neighbors.
4. Each of the 4 nearest neighbors contributes to the prediction based on its similarity to the test image. We assign weights to the neighbors based on their cosine similarity values divided by the temperature parameter τ. Higher similarity leads to higher weights. Let's assume our τ is set to 0.1.
5. Nearest Neighbor 1 (Bird image): Cosine similarity = 0.9, Weight =exp(0.9/0.1)=8.2
6. Nearest Neighbor 2 (Cat image): Cosine similarity = 0.7, Weight	=exp(0.7/0.1)=10.0
7. Nearest Neighbor 3 (Horse image): Cosine similarity = 0.6, Weight =exp(0.6/0.1)=16.2
8. Nearest Neighbor 4 (Cat image): Cosine similarity = 0.6, Weight =exp(0.6/0.1)=16.2
9. We sum up the weights of neighbors for each class.
10. Birds: Total Weight wBirds =8.2
11. Cats: Total Weight wCats=α2+α3=10.0+16.2= 26.2
12. Dogs: Total Weight wDogs=0 (no dog neighbors in this example)
13. Horses: Total Weight wHorses=α3=16.2
The class with the highest total weight is Cats. Therefore, the test image x^\hat{} is predicted to belong to the Cats class.













