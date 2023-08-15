# Paper Title: Deep Cluster for Unsuperervised Learning of Visual Features - FAIR
DeepCluster - The parameters of a neural network and the cluster assignments of the resulting features are jointly learned. With k-means it iteratively groups the features and uses the subsquent assignments as supervision to update the weights of the network. 

Clustering techniques have mostly been created for linear models using pre-set features. They struggle when features need to be learned at the same time. For instance, training a convolutional neural network with k-means could result in a simple solution where features become zero and clusters merge into one.

This research centers on training convolutional networks (convnets) end-to-end using a clustering method introduced by the authors. The process involves alternating between clustering image descriptors and adjusting convnet weights by predicting cluster assignments. While the primary focus is on k-means, other clustering approaches like power iteration clustering can be employed. The study demonstrates the robustness of this approach even when architecture changes.

For experimentation, the authors use ImageNet, a dataset with a unique image distribution known for its balanced classes and diverse content, such as various dog breeds. The model's performance is also tested on Flickr images, and it maintains state-of-the-art results even when trained on this diverse data distribution.

## Unsupervised Learning by Clustering

### Random features and their performance:
###### Random Initialization of Neural Network Parameters:
When we start training a neural network, we need to initialize its parameters, which are the weights and biases of the network's layers. The usual practice is to initialize these parameters randomly. At this point, the network doesn't "know" anything about the data it will be processing.

###### Producing Features with Randomly Initialized Network:
When we pass input data (like images) through this neural network with randomly initialized parameters (fθ), it processes the data and generates a set of features. These features are abstract representations that the network learns to extract from the input data.

###### Limited Effectiveness of Random Features:
Since the network's parameters are random and haven't been learned from any data, the features produced by this network might not be very meaningful or useful. In other words, they might not capture the important patterns or characteristics of the data.

###### Surprising Performance on Transfer Tasks:
Despite the random nature of the features produced by the network, it turns out that even using these random features for certain tasks can yield better results than what would be expected by chance. For example, if you take these random features and use them as input for a multilayer perceptron (a type of neural network used for classification), the resulting accuracy when classifying images on a task like ImageNet (a large-scale image classification dataset) is 12%. This accuracy is significantly higher than the expected random chance accuracy of 0.1%.

The point is that even without any meaningful learning, the structure of the neural network and its layers can still capture some underlying patterns or structures in the data, resulting in better-than-random performance on certain tasks. This surprising behavior is attributed to the convolutional structure of the network, which is designed to capture local and hierarchical patterns commonly found in images.

In other words, even without proper training, the structure of the neural network (especially in the context of convolutional networks for image data) can inherently capture some useful information from the input data. This is why, when we apply these random features to a downstream task like classification (using a separate classifier on top of these features), the performance is still better than what you'd expect by random chance. The convolutional structure of the network contributes to this behavior by leveraging its design to detect local patterns in the input data.

### Convolutional Structure and Weak Signal:
The convolutional structure of neural networks, like the AlexNet used in this case, introduces a strong prior on the input data. This structure is inspired by the way our brain's visual cortex processes information in a hierarchical and local manner. Despite the random nature of the initial parameters, this convolutional structure provides a weak signal that helps the network extract some meaningful features from the data.

When it's mentioned that the parameters (weights and biases) of the neural network are randomly initialized, it means that the network's ability to generate meaningful features from the data is quite limited initially. Randomly initialized parameters don't reflect any specific patterns in the data; they're essentially random values.

However, the reason why even these randomly initialized networks show some level of success is because of the underlying convolutional structure of the network itself. This structure is designed to capture local patterns and hierarchical features in the input data, particularly in the context of images.

While the random parameters don't provide a strong signal initially, the convolutional architecture itself imposes a certain prior or expectation on how patterns in the data might be detected. Think of this convolutional architecture as a template or a set of filters that are applied across the input data. These filters, although initialized randomly, are designed to detect edges, textures, shapes, and other simple features.

So, while the initial parameters are indeed random, the convolutional structure, guided by the principles of local feature detection, adds some level of structure to the randomness. This can allow the network to capture basic features even before any proper training occurs.

In a nutshell, the convolutional structure introduces some inherent biases that make even randomly initialized networks capable of extracting some useful information from the data, although this performance is far from what can be achieved with proper training.

### Exploiting Weak Signal with Deep Clustering:
The main idea of this work is to leverage the weak signal generated by the convolutional structure to improve the performance of the network. They achieve this through a technique called "Deep Clustering."

### Deep Clustering Approach:
The goal of Deep Clustering is to iteratively enhance the feature extraction process of the randomly initialized network. Here's how it works:
###### Step 1 - Feature Extraction:
The input data (images, in this case) is passed through the randomly initialized convolutional layers of the network, which produces a set of features for each image.
###### Step 2 - Clustering:
The features obtained from the previous step are grouped or clustered based on their similarities. This means that features that are similar to each other are grouped together.
###### Step 3 - Pseudo-Labeling:
The cluster assignments obtained in the previous step are treated as "pseudo-labels." These labels are not true ground-truth labels but rather indications of similarity based on the features. Each feature group is assigned a label based on its cluster.
###### Step 4 - Learning with Pseudo-Labels:
The network is then fine-tuned using these pseudo-labels. The objective is to minimize the negative log-softmax (multinomial logistic) loss, which is a common loss function for classification tasks. The idea is to adjust the network's parameters and possibly the weights to improve the alignment of the features with these pseudo-labels.
###### Step 5 - Iteration:
Steps 1 to 4 are repeated iteratively. With each iteration, the network's features become more aligned with the pseudo-labels, and the clustering process helps in refining the features further.

By following this deep clustering approach, the researchers aim to bootstrap the discriminative power of a randomly initialized convolutional neural network by gradually improving the features through the iterative process of clustering and fine-tuning based on pseudo-labels. This makes use of the convolutional structure's weak signal to enhance the network's performance on downstream tasks.

###### k-means Clustering: 
The k-means algorithm's goal is to group similar feature vectors into k distinct clusters. The algorithm learns k cluster centroids in a d-dimensional space, where d is the dimensionality of the feature vectors.

###### Problem Formulation: 
k-means jointly optimizes two components: the cluster assignment for each data point and the cluster centroids. The objective is to minimize the sum of squared distances between each data point and its assigned centroid, represented by the Euclidean distance.

###### Optimal Assignments and Centroids: 
Solving the k-means problem provides the optimal cluster assignments for each data point and the centroid matrix. The optimal assignments indicate which cluster each data point should belong to, and the centroid matrix represents the center of each cluster.

###### Pseudo-Labels: 
The obtained optimal assignments are treated as "pseudo-labels." These pseudo-labels are used to guide the training of the neural network, but the centroid matrix isn't actually utilized in this process.

###### Alternating Procedure: 
The DeepCluster method alternates between two main steps:
1. Clustering and Pseudo-Labeling: The features are clustered using the k-means algorithm, generating the pseudo-labels. This involves assigning each feature to the closest cluster centroid.
2. Updating Convnet Parameters: The neural network is updated by training it to predict the pseudo-labels generated in the previous step.

Overall, DeepCluster utilizes the k-means clustering algorithm to create pseudo-labels for the features extracted from the neural network. These pseudo-labels are then used to guide the network's training in an alternating manner, ultimately enhancing the representation learning process.

## Avoiding trivial solutions
Alternating procedures like the one used in DeepCluster can sometimes converge to trivial solutions.
###### Trivial Solutions in Discriminative Clustering: 
The issue of trivial solutions is not unique to unsupervised neural network training but applies to any method that simultaneously learns a classifier and assigns labels to data points in a clustering scenario. This can also be observed with linear models. Trivial solutions refer to instances where the algorithm allocates very few points to some clusters, making the problem too easy by heavily biasing towards underpopulated clusters. This can happen if the algorithm labels a few easy-to-classify points correctly, leading to an overly optimistic solution.

###### Challenges of Discriminative Clustering: 
The challenge lies in finding meaningful clusters that are well-distributed and representative of the data. While linear models are also prone to such solutions, they can be mitigated more easily due to their simplicity. However, neural networks are highly nonlinear and have a large parameter space, making it more difficult to ensure that the solutions are not trivial.

###### Addressing Trivial Solutions: 
Common strategies to address trivial solutions involve adding constraints or penalties to encourage more balanced cluster assignments. For example, one could add a term in the optimization process that enforces a minimum number of points per cluster. This would discourage the algorithm from forming tiny clusters with just a few points.

###### Challenge with Large Datasets: 
However, in the case of training convolutional neural networks (convnets) on large-scale datasets, applying such constraints or penalties over the entire dataset becomes impractical due to computational complexity.

### The solutions provided/used by researchers:
#### Empty clusters 
###### Empty Clusters and Causes:
In the context of discriminative models, such as those used in clustering or classification tasks, the algorithm's aim is to learn decision boundaries or groups that separate different classes or clusters. However, in some cases, the algorithm might arrive at a trivial solution where it decides to assign all input data points to a single cluster. This phenomenon is not specific to a particular model type; it can occur in both linear models and convolutional neural networks (convnets).

###### Optimal Decision Boundary:
Imagine you're working with a clustering algorithm, and its primary goal is to separate data points into different clusters based on certain criteria. Now, let's consider a scenario where the algorithm faces a choice: it can either create complex decision boundaries to separate points into multiple clusters, or it can decide to group all the points into a single cluster.

The counterintuitive part here is that from a mathematical optimization perspective, the second option—putting everything into one cluster—can sometimes be an "optimal" choice. This might seem strange because it doesn't look like a meaningful clustering, but mathematically, it serves the algorithm's primary goal, which is to separate points as effectively as possible based on its given criteria.

The main objective of the clustering algorithm is to group data points in a way that best aligns with its internal criteria. If the algorithm can achieve this objective by forming one big cluster and separating nothing, it might choose this option. This can happen when there are no specific mechanisms or constraints in place to discourage the formation of empty clusters.

Empty clusters occur when a clustering algorithm ends up with a cluster that doesn't have any data points assigned to it. This can happen for various reasons, including the algorithm's tendency to find the path of least resistance to satisfy its optimization goals.

In the absence of mechanisms or penalties that discourage the formation of empty clusters, the algorithm might find it convenient to assign all data points to a single cluster. This is because it's easier to satisfy the algorithm's primary goal this way, even though the result might not seem particularly meaningful in terms of grouping similar data points together.

###### Addressing Empty Clusters:
One workaround to address the problem of empty clusters, which can happen in both linear models and convnets, involves a trick commonly used in feature quantization (a process of reducing the number of distinct features). This workaround is particularly useful when employing k-means clustering.

1. Reassigning Empty Clusters: When a cluster becomes empty during the k-means optimization process, instead of letting it remain empty, a technique called "reassignment" is applied. In this approach, if a cluster becomes empty, the algorithm doesn't just leave it as such; instead, it takes measures to keep all clusters populated.
2. Random Selection and Perturbation: The reassignment process involves randomly selecting a non-empty cluster. Then, the centroid of this non-empty cluster is used as the new centroid for the empty cluster. A small random perturbation is often added to the selected centroid before using it as the new centroid. This perturbation introduces a bit of randomness to ensure that the clusters are not forced to be too similar to each other.
3. Reassignment of Data Points: After reassigning the centroid, the data points that were originally assigned to the non-empty cluster are then redistributed to the two resulting clusters – the previously non-empty cluster and the newly filled one.

###### let's go through an example to help clarify the process of reassigning empty clusters during the k-means optimization:
Suppose we have a dataset of points in a two-dimensional space. We're using the k-means algorithm to group these points into two clusters (k=2). The algorithm starts by placing two initial centroids randomly.
1. Initialization:
Let's say the initial centroids are:</br>
Centroid 1: (2, 3)</br>
Centroid 2: (5, 6)</br>
2. Iteration 1:
During the optimization process, the algorithm assigns each point to the closest centroid based on distance. However, let's imagine that all points are closer to Centroid 2. This might lead to all points being assigned to Cluster 2, leaving Cluster 1 empty.
3. Empty Cluster Issue:
Now, we have an issue: Cluster 1 is empty. This can happen due to the algorithm's tendency to find solutions that optimize its objectives, even if it means assigning all points to one cluster.
4. Reassignment:
To address this empty cluster problem, the reassignment technique comes into play. Instead of leaving Cluster 1 empty, the algorithm follows these steps:
  1. Randomly select a non-empty cluster, let's say Cluster 2.
  2. Use the centroid of Cluster 2 (e.g., (5, 6)) as the new centroid for Cluster 1.
  3. Add a small random perturbation to the selected centroid to introduce some variation.
5. Data Redistribution:
After the centroid of Cluster 1 is reassigned, the data points that were originally assigned to Cluster 2 need to be redistributed. They are split between the two clusters, with the reassigned Cluster 1 and the original Cluster 2.

The small random perturbation added to the selected centroid helps prevent the reassigned cluster from being too similar to the original one. This perturbation ensures diversity among clusters and avoids the risk of creating identical clusters.

### Trivial Parametrization
This happens when a majority of the data points are assigned to a few clusters, leading the network's parameters to become specialized in distinguishing only those particular clusters. In extreme cases, where most clusters contain only one data point (singleton clusters), the optimization process can lead to a situation where the network behaves predictably for those clusters but ignores the input data.

This issue is similar to problems encountered in supervised classification when there's a severe imbalance in the number of samples per class. For instance, in a scenario where only a few classes have a significant number of samples while others have very few, a classifier might become overly biased towards predicting the dominant classes, effectively ignoring the others.

Imagine you are training a neural network to perform clustering on images of animals. The goal is to group similar animals together in clusters. Each cluster should represent a different type of animal, like dogs, cats, birds, etc.

During training, the neural network learns to group the majority of images into only a few clusters. For instance, let's say the network tends to put all the dog images into a single cluster, and all the cat images into another. These clusters become "dominant" because most of the training data points are assigned to them.

Due to this clustering pattern, the network's parameters become specialized in distinguishing between these dominant clusters, particularly dogs and cats. The network gets really good at telling dogs apart from cats.

As a result, when the network encounters new images during inference (testing), it becomes overly focused on detecting whether an image is a dog or a cat. It might ignore other important features of the images, such as their overall appearance or context. This is because the network's parameters have become biased towards making dog-versus-cat distinctions.

In an extreme case, let's say there are some clusters that contain only one image each. These are called "singleton clusters." The network could end up completely ignoring these clusters because it has specialized itself so much in distinguishing dogs and cats.

So, essentially, this phenomenon means that the network becomes so good at distinguishing the dominant clusters that it starts ignoring the broader context and diversity of the input data. This is problematic because the network's purpose is to learn meaningful and general features from all types of data, not just the most common ones.

To address this issue in the context of deep clustering, a strategy is introduced to ensure that the network doesn't become overly specialized in a small set of clusters. The approach involves the following steps:
1. Sampling Strategy: Instead of treating all clusters equally, we might choose to sample images from different clusters in a balanced way. This ensures that the network doesn't disproportionately focus on the most common clusters.
2. Weighted Contribution: In the loss function that guides the training process, each data point's contribution to the loss is not equal. The contribution is adjusted based on the size of the cluster to which that data point belongs. Data points from smaller clusters are given more weight, and data points from larger clusters are given less weight. This way, the network is encouraged to pay more attention to smaller clusters, ensuring that it doesn't ignore them in favor of the larger ones.

Suppose we have three clusters:</br>
Cluster A: 1000 data points</br>
Cluster B: 500 data points</br>
Cluster C: 100 data points</br>
If we were to naively train without any sampling strategy, the network might become biased towards Cluster A, as it has the most data. But if you use a sampling strategy:
1. Uniform Sampling: We randomly choose an equal number of data points from each cluster during each training iteration. For instance, you might sample 100 data points from Cluster A, 100 from Cluster B, and 100 from Cluster C. This ensures that the network gets exposed to examples from all clusters in a balanced manner.
2. We use the cluster assignments as a basis for selecting data points for training. The idea is to ensure that each cluster contributes proportionally to the training process, regardless of the cluster size. This helps prevent the model from becoming biased towards larger clusters.
This means that you want to make sure that, during training, you see approximately equal numbers of data points from each cluster relative to their sizes.
1. We have a total of 1600 data points (1000 from A + 500 from B + 100 from C).
2. We want to sample, say, 300 data points for each training iteration.
3. We determine the proportions based on the cluster sizes: A contributes 1000 / 1600 ≈ 0.625, B contributes 500 / 1600 ≈ 0.3125, and C contributes 100 / 1600 ≈ 0.0625.
4. During each training iteration, we would then sample approximately 187 points from Cluster A (0.625 * 300), 94 points from Cluster B (0.3125 * 300), and 19 points from Cluster C (0.0625 * 300).
This way, we are effectively ensuring that each cluster contributes to the training process in proportion to its size. It's not about achieving a perfectly equal number of data points from each cluster, but rather about ensuring that each cluster's impact on training is balanced relative to its size. This helps in preventing the network from disproportionately focusing on larger clusters and promotes a more fair and generalized learning process.


