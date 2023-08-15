# Contrastive Loss
Contrastive loss measures the similarity between positive pairs of samples while maximizing the dissimilarity between negative pairs. In recent years, pre-trained models that employ contrastive loss have shown impressive outcomes and have even attained leading positions in self-supervised representation learning. However, the precise workings and efficacy of this loss function are still being actively investigated. Various research endeavors are dedicated to exploring the theoretical, experimental, and geometrical dimensions of this loss, aiming to unveil its complexities.
</br>
## <span style="color:blue"> Paper Title : [Tian et al., 2020](https://arxiv.org/abs/2005.10243) - What Makes for Good Views for Contrastive Learning? </span>

Understanding the importance of invariant features from a story 'Funes the Memorious' by Jorge Luis Borges.</br>
InfoMax principle.</br>
InfoMin principle.</br>
Optimal views for contrastive representation learning are task-dependent.</br>
Reverse U shaped relationship between an estimate of mutual information and representation quality.</br>
Semi-supervised method to learn effective views for a given task.</br>

SimCLR is one of the famous approaches using contrastive loss. It maps images to a lower dimensional space and that's our embedding vector or trained representations. We want representations such that two different crops of same image should be close as much as possible i.e. they should be attracted to each other and two crops from different images should be repel each other. So different parts of the same image get represented alike while of different images end up away from each other in the embedding space. Two different views can be crops (disjoint crops or one crop is subset of another crop), color channels, etc.</br>

This idea is not new, this goes back to 1992 with the paper titled 'Self-organizing neural network that discovers syrfaces in random-dot stereograms' by Hinton and Becker. This is essentially SimCLR but without deep networks.</br>

So, to set the goal clear - the objective is to learn an embedding that pulls positive pairs together and pushes negatives apart, where positive pairs are two views from the same image and negatives pairs are two views from different images. The numerator in the loss function is all about trying to increase the similarity of two positive views or a positive pair. The denominator is all about forcing two views of different images or negative pairs to map to different points in embedding space, i.e., they are to be pushed apart.</br>

###### InfoNCE Loss
![InfoNCE Loss](https://github.com/Varchita-Beena/SSLHavenCorner/blob/SSLIncoming/Images/UCL_W1_EQ1.png)
InfoNCE loss was introduced in the paper with title Contrastive Predictive Coding.</br>

Isn't the concept of memory, perception and the complexities of human consciousness is amazing? 'Funes the Memorious' a short story by Jorge Luis Borges explores these concepts. What happens if we have perfect memory, is it a curse or boon? The Funes character gains an extraordinary mental ability. He becomes 'memorious', meaning he could remember every single detail of his life in the most vivid and precise manner. He remembers the most trivial details such as exact shape and position of clouds on a particular dat. He can recall every leaf of a tree and every nuance of a conversation he has had. His memory is so overwhelming that he can't percieve geenral ideas or concepts. Instead of thinking in abstract terms, ge saw the world as a sea of individual and specific details. He can't engage in meaningful conversation, read literature, or engage in creative thinking. His memory is so complete that it hampers his ability to make connections and derive meaning from his experiences.</br>

Funes becomes bothered that a “dog at three fourteen (seen from the side) should have the same name as the dog at three fifteen (seen from the front). Well it seems excess of anything is really dangerous. Our mind builds representations of identity that discards the details that are not needed. Our mind makes view-invariant representations and these are important type of representations in research on multiview coding. As discussed already contrastive multiview learning is all about pushing two views of the same image towards each other as possible, and different views of different images should be pulled apart. This means we have to remember a lot of details but at the same time we have to discard details that are not relevant in making the representations abstract. Now the important question is 'which viewing conditions should we be invariant to?'. We can not use representations invariant to time when our task is to classify the images by the tiome of the day. If we remember everything like Funes i.e., each specific viewing angle is having a different representation then it won't be able to tract a dog as it moves in the video/image.</br>

Hence, the representations should be abstract (invariant) in the right amount to be helpful for downstream tasks i.e., not excess information (so invariant that it can recognize the same dog) and not to discard a lot of information that they won't be helpful on downstream tasks. The focus is to capture the shared information between two views in the contrastive learning, hence the choice of views is very important. Views can be - crops, disjoint crops, small subset of the big crop, over-lapping crops, different color channels, slices in time etc from the same data point. How can we achieve a balance of viewpoints that provide precisely the necessary information without excess or deficiency?</br>

The authors of this paper do this in two ways:
All about the downstream tasks:- If we know the downstream tasks, it's possible to design the effective views. </br>
Sweet spot:- Empirically they show that for common view generation methods, there exists a sweet spot in terms of downstream performance where the mutual information (MI) between views is neither excessive nor insufficient. </br>

This paper uses labeled data to learn better views but still perform contrastive learning using only unlabeled data.</br>

## InfoNCE loss
Let's discuss some points on the equation described at the start for InfoNCE loss. We can say that the positive images come from a joint distribution over both views p(v1, v2) and the negative pairs from a product of marginal p(v1)p(v2). So the goal of the contrastive learning is get an estimator of the mutual information between v1 and v2 that discriminates between samples from the empirical joint distibution p(v1)p(v2|v1) and the samples from the product of marginals p(v1)p(v2).</br>

Is it shown that the InfoNCE loss is the lower bound on the information shared between the raw views I(v1;v2). I(v1;v2) >= I(z1;z2) >= InfoNCE loss. z1 and z2 are the embeddings that we get by applying some model/operations on the raw data. The simple understanding for this inequality is through- recall data processing inequality. It says that when we apply some local physical operation on signal then the information content of a signal cannot be increased via a local physical operation. It's true here also as we are applying convolutions on images so the information won't increase after every operation. Hence, I(v1;v2) >= I(z1; z2) >= log(K) - InfoNCE loss. K is the number of negative samples. This says that, if we are to minimize the InfoNCE loss, then as a result we are maximizing the information content between the raw images and the more negative images we have, the more our model will work hard to find a good solution and hence InfoNCE is the lower bound. </br>

Create views: Randomly crop two patches from the same image with various offsets. We are increasing the spatial distance between two patches. After contrastive training stage, evaluate in dataset by freezing the encoder and training a linear classifier. So the plot is mutual information vs accuracy. The result is reverse-U shaped. So the InfoNCE loss is reduced, hence we are increasing the mutual information and the downstream task accuracy first increases and then decreases. When we have the patch disatnce very high we are not getting very good accuracy as the captured shared information is very less. Even when we have the least spatial distance we are not able to get very fine accuracy as it captures a lot of shared information. There exists a sweet spot where we are able to capture right amount of shared information that is enough for downstream task. The plot is about the patch distances. Paper shows experiments for color space, color jittering, random resized crop. For all they are able to get reverse U shaped curve. </br>

We can see the trade off between how much information our views share about the input and how well learned representations performs at predicting y for a task. When we are not able to achieve desired accuracy for downstream task it implies that we are missing on some task-relevant information and we are the needed information is being discarded by the view, degrading the performance.</br>

When we are capturing a lot of information between the views we are landing in the excess noise zone as we are capturing information beyond need. We are gathering some noise or background, decreasing the transfer learning accuracy, hence the worse generalization. </br>

Sweet spot is all about capturing right amount of information, no noise, only relevant information for the downstream task.</br>

We often do not have access to a fully labeled train set that specifies the downstream task in advance, and thus evaluating how much task-relevant information is contained in the views and representation at training time is challenging. The construction of views has typically guided by domain knowledge that alters the input while preserving the task-relevant variable. </br>

Toy dataset - mixes three tasks. 
Moving-MNIST: Videos where digits move inside a black canvas with constant speed and bounce off of image boundaries. 
STL-10: A fixed background image
Final dataset - Colorful Moving MNIST: it has three factors of variation in each frame: the class of the digit, the position of the digit and the class of the background 
View-1 : sequence of frames containing moving digit
View-2 : (single image) The positive view is sharing some factor with sequence and negative doesn't share any factor. 
3 downstream tasks are considered for an image: predict the digit class, localize the digit and classify the backhround image (10 classes from STL-10). Authors are freezing the backbone and training a linear task-specific head.

Single Factor Shared: The performance is significantly affected by what is shared between view-1 and view-2. when both only share background then contrastive learning can hardly learn representations that capture digit class and location. 

Multiple Factors Shared: Experiments show that one factor can overwhelm another, when background is shared, latent representation leaves out infromation for discriminating or localizing digits. May be because the information bits of background predominates and the encoder chooses the background as shortcut to solve the contrastive pre-training task. When they share digit and position, the digit predominates.

Authors use flow-based models to transform color spaces into novel color spaces. These transformations result in distinct views obtained by splitting color channels. These views are then subjected to contrastive learning and linear classifier evaluation, with a focus on maintaining the properties of color spaces during transformation. The experiments are conducted using pixel-wise operations and different types of flows, and the evaluation is performed on the STL-10 dataset.

For the above purpose, authors use adversarial training. They use "generator" that changes how pictures look. They also use "encoders," that try to figure out if the changed pictures are different or similar. The generator tries to make the changed pictures look similar, while the encoders try to tell them apart. This way, the generator learns to make pictures that are different but still look similar. The goal is to find a good balance between changing the pictures and keeping them similar. This helps the generator create useful and meaningful changes in the pictures. They use a specific method to measure the success of this process, and they make sure that the generator doesn't come up with meaningless or weird changes.

Experiments are done using different color representations like RGB and YDbDr. They notice the mutual information measure (INCE) and the accuracy of the tasks they're testing on follow a curve that looks like a reverse "U" shape. What's especially intriguing is that the YDbDr color representation is already close to the optimal point they are aiming for. This aligns with their belief that the way colors are separated into brightness and color information is a good strategy for capturing essential details while still making objects recognizable.

They also mention another color decomposition method called Lab, which performs similarly well. They point out that this method was designed to mimic how humans perceive color, which suggests that human color perception might actually be quite effective for self-supervised learning. 

However, they also note a challenge in their approach. The training process, which is similar to a type of machine learning called GAN, can be unstable. This means that different attempts with the same settings can lead to different results. They believe this instability might be because the view generator doesn't know anything about the final tasks they're testing on, so some constraints are not met perfectly.

To address this challenge, they propose a new method that combines both unsupervised and semi-supervised learning to improve stability and performance.

Semi-supervised view learning.
A method that leverages the availability of a small number of labeled examples for the downstream task. They want to guide the view generator, represented by "g," to retain the information about the labels. To achieve this, they introduce two classifiers, denoted as c1 and c2, which help in performing classification during the process of learning views. The goal is to optimize an equation that involves these classifiers and the mutual information measure (INCE). 

The INCE term applies to all data (both labeled and unlabeled), while the classifiers' terms are specific to the labeled data. In each iteration, they take both an unlabeled batch and a labeled batch and use the frozen view generator to create views for the unsupervised contrastive representation learning.

They show that, regardless of the original color space and whether the generator operates with volume-preserving (VP) or non-volume-preserving (NVP) flows, the learned views tend to be centered around the optimal performance region, or the "sweet spot." This outcome highlights the importance of incorporating information about the labels.

To further analyze their approach, they compare different types of view generators: "supervised," "unsupervised," and "semi-supervised" (a combination of supervised and unsupervised losses). They also include the baseline of using contrastive learning over the original color space, referred to as "raw views." The semi-supervised view generator significantly outperforms the strictly supervised one, which underscores the value of reducing mutual information between the learned views. 

They also compare the performance of their approach, g(X), with the raw input data X (which is either RGB or YDbDr) using larger backbone networks. This comparison shows consistent improvement using the learned views over the raw input. This demonstrates the effectiveness of their method in enhancing the representation learning process.








