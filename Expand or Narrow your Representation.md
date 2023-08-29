# Paper: [A surprisingly simple technique to control the pretraining bias for better transfer: Expand or Narrow your representation](https://arxiv.org/abs/2304.05369)

## Outline
1. Introduction
2. Information bottleneck
3. A linear projector
4. Linear projector with unbalanced sampling
5. Linear projector with balanced sampling
6. Nonlinear projector with unbalanced sampling
7. Nonlinear projector with balanced sampling
8. Number of parameters
9. Wider SSL representations are sparse
10. Wider representation are more easily linearly separable
11. Wider representations are binarizable
12. Summary

## Introduction
Pretext task is very important for self-supervised learning to learn representations, but pretext task and downstream tasks are very different, hence there is pretraining bias present in the model. Pretraining bias – a tendency for the representations to become specialized for the pretext task and potentially not generalize well to other tasks.

When SSL models are pretrained on a pretext task, they learn to capture certain patterns or features in the data that are useful for solving the pretext task. However, these patterns or features may not necessarily be the most relevant or important for the downstream tasks that the model will eventually be applied to.

This pretraining bias can affect the performance of the SSL model on downstream tasks. The model's representations might be more suited for the pretext task and might not generalize well to the specific requirements of the downstream tasks. In other words, the model might struggle to effectively leverage the learned representations for tasks that it was not explicitly pretrained on. 

Addressing pretraining bias is a key challenge in SSL and transfer learning, and researchers work to develop methods that can mitigate this bias and allow the model's pretrained representations to be more adaptable and useful across a wider range of tasks.

###### projector
A common technique used to enhance the robustness of deep networks to such bias is the addition of a small projector on top of the backbone network during training. This projector, often implemented as a 2 or 3-layer multi-layer perceptron, is designed to transform the features extracted by the backbone network into a more suitable representation for the downstream tasks. However, this method only partially mitigates the bias. Some methods like SimCLR achieve similar results whether they use a small or large projector dimension, while other methods like VICReg are sensitive to the dimensionality of the projector embeddings.

###### Cluster priors
The pretraining bias can still negatively affect downstream task performance, especially on datasets that are imbalanced, meaning some classes are much more frequent than others. This poor performance might be because SSL methods implicitly assume that all classes have equal importance, which is not the case in real-world datasets. To tackle this issue, researchers have explored ways to correct the implicit bias in SSL methods. 

For instance, one type of SSL method involves clustering data samples, and a study has proposed a modification to their clustering-based SSL approach (called MSN) to account for the class distribution better. They adjusted the cluster priors to align more closely with the actual class distribution, thus addressing the bias problem. However, this modification is applicable only to SSL methods based on explicit clustering. 

Cluster priors are probabilities assigned to each cluster that indicate how likely a data sample belongs to that cluster. By modifying these priors, the researchers aimed to make sure that the clustering process takes into account the actual distribution of classes in the dataset. In simpler terms, they adjusted the way clusters are formed so that they better match the real distribution of classes.

###### Hidden Uniform Prior:
A study on "hidden uniform prior in self-supervised learning" has brought attention to a concept called the "hidden uniform prior" in the context of training contrastive self-supervised models.
 
The term "hidden uniform prior" refers to an underlying assumption or tendency that exists within contrastive self-supervised learning methods. This prior implies that the models treat all data samples uniformly or equally during the learning process, regardless of their actual class distribution or prevalence.

Link to K-means Clustering: The researchers connect this hidden uniform prior to K-means clustering, a method used in unsupervised learning to group similar data points together. They suggest that many self-supervised learning methods, including contrastive ones, exhibit a tendency to cluster data in a uniform or balanced manner in the representation space, influenced by this hidden uniform prior.

Harmful on Imbalanced Data: Paper shows that this hidden uniform prior can be problematic, particularly when dealing with imbalanced datasets. Imbalanced datasets have significantly more instances of some classes than others. When the hidden uniform prior is applied, it can lead to an inappropriate representation of the dataset, ignoring the class imbalances and resulting in suboptimal performance on downstream tasks.

Dimensional Bottleneck: In the current paper being discussed, the authors argue that the most significant issue arising from this hidden uniform prior is its interaction with the "dimensional bottleneck." The dimensional bottleneck refers to situations where the dimensionality (complexity) of the model's representation is constrained or limited. When this bottleneck is too strong, it forces the model to focus on capturing certain features that might not align well with the specific tasks the model will later perform.

###### Sparse Representations
There's an interest in creating sparse representations of data. This means that only a small number of parts or features are used to describe each piece of data. Methods have been developed to achieve this, where certain techniques are used to make most parts inactive. Methods belonging to this introduce constraints or specific training rules to ensure only a small portion of features are active.

However, the current paper looks at a different perspective. It shows that in the context of self-supervised learning (SSL), where models learn by understanding data relationships, sparse representations can naturally emerge. This means that many parts of the data aren't actively used to describe it.

Particularly, when using an activation function called Rectified Linear Unit (ReLU), these naturally sparse representations become even sparser. This happens without needing any special rules or constraints during training.

The paper points out that when using wider models, which means models with more components, this sparsity becomes even more pronounced. In some cases, more than 80% of the components don't contribute much to the representation.

Interestingly, these naturally sparse representations obtained through SSL can be turned into a binary form, which means each component is either 0 or 1. This is useful because binary data takes up less space and is computationally efficient.

The paper states that using these sparse binary representations doesn't significantly hurt performance. In other words, even though a lot of components are inactive, the representation is still useful for various tasks.

###### Current paper
Here the authors are focusing on a different approach to mitigating pretraining bias. Instead of modifying the projector architecture, they propose a simpler and less explored method: changing the dimensionality of the backbone's last block. The "backbone" refers to the core architecture of the neural network responsible for extracting features from input data. This last block, which is a part of the backbone, is typically responsible for transforming extracted features into a higher-level representation before the final predictions. 

The authors show that by adjusting the dimensionality of this last block (essentially changing the size of this transformation step), the pretraining bias can be effectively reduced. This adjustment influences the kind of information that is retained and conveyed by the backbone representation. By altering the size of the last block, the representation's capacity to capture relevant features for downstream tasks is enhanced.

This simple yet overlooked technique of changing the dimensionality of the backbone's last block is shown to be highly effective in improving the performance of pretrained models on various downstream tasks, both for models pretrained through self-supervised learning and those pretrained through supervised learning. The advantage of this approach lies in its simplicity and efficiency, offering a practical solution to addressing pretraining bias and enhancing the transferability of learned representations. This also significantly improve transfer performance, as well as robustness when training on datasets with long-tailed class distribution.

A recent study observed that using a larger backbone representation (wider features) improves the performance of linear probing techniques in CISSL (a particular SSL method). The authors of the current paper extend this observation to several other SSL methods like SimCLR, VICReg, and Byol, as well as to the supervised setting. They emphasize that having wider representations in the backbone is crucial for reducing the bias introduced during pretraining.

The authors not only provide quantitative analysis (measuring performance) but also qualitative analysis (understanding the nature of improvements) of how larger representations impact various downstream tasks and different pretraining setups.


## Information bottleneck
We have deep network backbone function (f) that takes an image as input and then transforms it into a representation in a space. This representation from the backbone is further processed by a projector function (g). The projector maps the higher-dimensional representation to a lower-dimensional space.

## Linear projector
The learning process involves considering a batch of images, along with their augmented versions. These batches are collectively represented as X. The loss function L is optimized based on this batch, with the goal of finding the best representations that align well with the SSL objectives. We thus optimize L(g(f(X))).

In this specific scenario, the projector is a linear function represented by a matrix W. Each row of this matrix transforms a part of the higher-dimensional representation to a corresponding part in the lower-dimensional space.

With the linear projector in place, the loss function L is computed by applying the projector (W) to the backbone representations (f) of the batch of images (X). This gives a lower-dimensional representation.

The training process involves optimizing this loss. To do this, gradient signals are computed and backpropagated. The gradient signal backpropagated to the backbone representation (f) is given by the product of the gradient of the loss with respect to the lower-dimensional representation and the transpose of the linear projector matrix

In simple terms, this process involves taking the high-dimensional image representations, transforming them into a lower-dimensional space using a linear projector, and then computing how changes in the projector's parameters affect the final loss through the backpropagation of gradients. The idea is to learn the best projection that aligns the lower-dimensional representations with the objectives of SSL, despite the inherent differences between SSL objectives and real-world data.

The backpropagation process is constrained to a subspace whose maximum dimension is determined by the rank of the linear projector matrix (W), which is at most min(D, K), where D is the dimension of the original representation and K is the dimension of the projected representation.

To control the amount of gradient-based information that flows through the loss, it's necessary to have a scenario where the original dimension (D) is significantly larger than the projected dimension (K). This creates what's called an "information bottleneck." Essentially, having D >> K ensures that the influence of the loss on the representation is limited due to the constrained subspace.

Imagine we have a machine learning problem where we're trying to classify different types of fruits based on their features. Each fruit is described by several attributes like size, color, and texture. We want to build a model that takes these attributes as inputs and predicts the type of fruit.
Let's say we have 10 different attributes to describe each fruit. This means our input space is 10-dimensional (D = 10).
We decide to use a linear projector (W) to transform these attributes into a lower-dimensional space for better representation. For instance, we want to reduce the 10 attributes down to 3 dimensions (K = 3).
The transformed 3-dimensional attributes are further processed by a linear backbone representation (f), which further extracts meaningful patterns from these 3 dimensions.
Now, let's consider the case where D (the original attribute dimensions) is greater than K (the projector dimensions):
When we perform the linear projection with W, you are effectively selecting a specific subspace in the 10-dimensional attribute space that aligns with the directions defined by W. This subspace has 3 dimensions (K).
The remaining dimensions (7 in this case) are orthogonal to the subspace aligned with W. This means they are perpendicular to the directions defined by the projector W.
(Imagine we have a 3D space where you're working with vectors. Let's say we have a linear projector W that reduces the dimensionality from 3D to 2D. So, we have a 3x2 matrix W that transforms our 3D vectors into 2D vectors.
Now, let's consider a specific 3D vector, let's call it v, which has three components: v = [v1, v2, v3]. When we multiply this vector with the matrix W (i.e., Wv), we get a new 2D vector with two components.
What the statement "remaining dimensions are orthogonal to the subspace aligned with W" means is that the third component of vector v (v3) and any other component beyond the first two (e.g., v4, v5, etc.) do not contribute to the transformation defined by the projector W.
In other words, the first two dimensions of the transformed vector (result of Wv) are aligned with the two columns of the matrix W. These dimensions are directly influenced by the action of the projector. However, the third dimension of the transformed vector (let's call it w3) and any other dimensions beyond it are not affected by the projector at all. They are "orthogonal" or "perpendicular" to the transformation defined by W.
So, when we apply the linear projector W to our original vector v, we get a transformed vector that retains the influence of only the first two components of v and discards any influence from the third component (and beyond) because they are orthogonal to the transformation.
The use of the term "orthogonal" here is an analogy to the geometric concept of vectors being perpendicular to each other in a Euclidean space. In linear algebra, when two vectors are orthogonal, it means they are independent or unrelated to each other. In the context of this discussion, the third dimension and the dimensions beyond it are not influenced by the transformation defined by W, and thus they are conceptually independent of the dimensions that are directly influenced by W.
The term "orthogonal" here is not referring to a literal geometric orientation, but rather to the independence of these dimensions from the action of the projector.)

Now, when we train our model using a learning algorithm, we're minimizing the loss by adjusting the parameters of your model (W and f) based on the training data.
The gradient signal from the loss, when backpropagated, will mainly affect the subspace aligned with W. Changes will happen along the directions that W defines.
However, the orthogonal subspace remains relatively untouched because the gradient signal can't move across the orthogonal directions.

This phenomenon is similar to trying to push a door open while it's already wide open. The direction in which we push is aligned with the door's movement (similar to the subspace aligned with W), while pushing perpendicular to the open door doesn't change its position (similar to the orthogonal subspace).

The key point here is that, in situations where we have more dimensions in our input attributes (D) than in our reduced representation (K), the gradient-based updates driven by the loss primarily affect the directions defined by the projector (W), leaving other directions relatively unaffected. This understanding helps in designing strategies to manage and control the flow of information during the training process for better model performance.

Now, consider if the linear projector (W) is kept fixed, and only the matrix V is trained from a random initialization. As K becomes smaller, a larger portion of the subspace of V will remain unaffected by the training process, resulting in the representation retaining more information about the input X without being significantly impacted by the SSL loss L.

In simple terms, the research explores how the relationship between the dimensions of the original representation, the projected representation, and the linear projector can affect the flow of gradient-based information during training. It's found that having a larger original dimension than the projected dimension can limit the impact of the SSL loss on the representation, leading to better performance in downstream tasks. This is because when the original dimension is much larger, there is more room for the gradient-based information to flow during training, which allows the self-supervised learning loss to have a greater impact on the representation. This, in turn, can result in representations that are more aligned with the downstream tasks and therefore lead to improved accuracy in those tasks. This concept was validated through experiments involving various architectural modifications.

"Dimensions of the original representation" refers to the output of the backbone network (also known as the feature map or representation obtained from the initial layers of the network)
The "projected representation" refers to the output of the projector that transforms the original representation to a different space. And "linear projector" refers to the combination of the linear projection applied on top of the original representation by the projector.

## Linear projector with unbalanced sampling
1. In this experiment, the researchers aim to investigate how the ratio of the original dimension (D) to the projected dimension (K) impacts the robustness of the linear projector to the uniform prior bias. To do this, they design an experiment inspired by a previous work that involved changing the sampling distribution during training. Instead of uniformly sampling images from the CIFAR10 dataset, they sample images belonging to only two different classes for each mini-batch.
   For instance, if we have a dataset with 10 classes (A, B, C, ..., J), during one mini-batch, the researchers might only sample examples from class A and class B. This means they are not considering any examples from classes C through J for that particular mini-batch. In the next mini-batch, they might sample examples from classes C and D, and so on.
2. They use a ResNet50 neural network architecture and apply the SimCLR (contrastive learning) criterion for self-supervised pre-training. In this experiment, they vary the dimensions of both the backbone representation (D) and the linear projector (K) while keeping the architecture of the backbone and projector the same. They then evaluate the performance of the representations on a downstream linear probe task.
3. Effect of D >> K: When the original dimension (D) is significantly larger than the projected dimension (K), the performance improves. This aligns with the concept that a larger D relative to K creates an information bottleneck that mitigates the impact of the uniform prior bias.
4. Effect of Decreasing K: For a given backbone dimension D, decreasing the projector dimension K leads to improved performance. This suggests that reducing the dimensionality of the projected representation (embedding) helps in mitigating the bias.
5. Effect of Increasing D: For a given projector dimension K, increasing the backbone dimension D also results in higher gains in performance. This reinforces the idea that a larger original dimension helps in countering the uniform prior bias.
6. Optimal Configuration: The figure demonstrates that the best performance is achieved when the original dimension D is very large and the projected dimension K is small. This indicates that a combination of a high-dimensional original representation and a low-dimensional projected representation is particularly effective in addressing the uniform prior bias and improving downstream task accuracy.</br>
![linear projector unbalanced sampling](https://github.com/Varchita-Beena/SSLHavenCorner/blob/SSLIncoming/Images/linear_proj_unbalanced_sampling.png)

## Linear projector with balanced sampling
1. In this phase of the experiment, the researchers transition to a balanced sampling scenario where images are uniformly sampled from the CIFAR10 dataset.
2. Projector Dimension: In the balanced sampling setup, changing the dimension of the linear projector (K) doesn't appear to bring significant benefits. This means that for a balanced dataset, the impact of varying the projector's dimension on the performance is not as pronounced.
3. Backbone Dimension: However, increasing or decreasing the backbone dimension (D) still has a notable effect on the performance. Increasing the size of the backbone while keeping the projector dimension fixed leads to an increase in accuracy.
4. Large-Scale Experiment on ImageNet: To further validate their observation, the researchers conducted an experiment using the larger ImageNet dataset. They fixed the backbone dimension at 32768 and varied only the projector dimension. The results show that decreasing the size of the linear projector improves the validation accuracy on ImageNet.
5. Decreasing Projector Dimension: The results demonstrate that decreasing the size of the linear projector's dimension while maintaining a fixed backbone dimension can lead to improved accuracy on ImageNet.
6. Efficiency Note: It's interesting to note that even with a relatively small number of dimensions (32) in the linear projector, a SimCLR model achieved a substantial 67.4% accuracy on the challenging ImageNet dataset.
7. In summary, the experiment conducted in this phase focuses on balanced sampling scenarios and observes that while changing the projector's dimension has limited impact, varying the backbone's dimension still influences accuracy. Decreasing the size of the linear projector seems to be an effective strategy for improving accuracy on large-scale datasets like ImageNet.
![Linear projector balanced sampling](https://github.com/Varchita-Beena/SSLHavenCorner/blob/SSLIncoming/Images/linear_proj_balanced_sampling.png)

## Non-linear projector with unbalanced sampling
1. Shift to Non-linear Projector and Unbalanced Sampling:
Following the insights gained from the linear projector experiments, the researchers now explore the impact of a non-linear projector on the performance of SimCLR when trained on unbalanced data. They continue using the CIFAR10 dataset, but instead of a linear projector, they employ a two-layer non-linear projector. The first layer maps the representation to K dimensions, while the second layer maintains a fixed dimensionality of 256.
2. Changing D/K Ratio: Unlike the results from the linear projector experiments, altering the ratio D/K while keeping D constant doesn't seem to bring any notable benefits in the non-linear projector scenario. In other words, varying the relative dimensions of the backbone and the projector doesn't significantly impact performance. Increasing D for Fixed K: However, consistent with previous findings, increasing the value of D (backbone dimension) while keeping K constant leads to substantial gains in accuracy.
3. Effect of Non-linearity: The experiment demonstrates that, unlike the linear projector experiments, adjusting the ratio D/K doesn't seem to have a clear impact on performance in the case of a non-linear projector.
4. Backbone Dimension: Increasing the size of the backbone representation (D) while maintaining a fixed dimension for the non-linear projector (K) continues to result in noteworthy accuracy improvements.

## Nonlinear projector with balanced sampling
In this section of the paper, the researchers conducted a series of experiments to explore the impact of using wider representations in traditional self-supervised learning (SSL) settings. They focused on the relationship between the size of the representation and the performance of different SSL methods.

###### Experiment on ImageNet Validation:
1. They trained various SSL models (SimCLR, VICReg, Byol) and a supervised model on ImageNet dataset while changing the dimension of the backbone representation (D) while keeping the projector fixed.
2. When using traditional supervised training, they observed that the performances on ImageNet improved when the backbone dimension (D) became smaller.
3. However, when considering SSL methods, they noticed a significant performance boost when using larger backbone dimensions (D).
4. This experiment highlighted a limitation in the current SSL literature, where SSL methods often use the same backbone architectures designed for supervised learning. The researchers hope their study will encourage the exploration of SSL-specific architecture designs that leverage the advantages of wider representations.

###### Evaluation on Downstream Tasks:
1. The researchers extended their evaluation to a wider range of downstream tasks, including ImageNet 10%, CIFAR10, CLEVR, Places205, and Eurosat.
2. They found that increasing the dimensionality of the backbone representation (D) benefited each of these datasets when using SSL methods.
3. An interesting observation was made regarding the performance of supervised and pretrained models on ImageNet and other datasets.
4. For supervised models evaluated on ImageNet itself (in-distribution), smaller D performed better because the model learned specific task-related invariances.
5. On the other hand, when evaluating models pretrained on ImageNet but tested on different datasets (out-of-distribution transfer), larger D yielded better results. This was because the invariances learned during pretraining could negatively affect generalization to different tasks.

In essence, the researchers showed that adjusting the ratio of D to K (where K is the projected dimension) in the SSL framework, by using wider representations, helps mitigate the pretraining bias and enhances the models' performance on various downstream tasks. This insight can lead to better design choices for SSL-specific architectures, allowing for improved generalization and adaptability across different tasks.

The researchers conducted a qualitative visualization experiment to further investigate their hypothesis about the impact of wider backbone representation sizes in the context of self-supervised learning (SSL). They used a method called Representation Conditional Diffusion Model (RCDM) to map SSL representations back to the image space and analyze the information retained in the representations.

A Representation Conditional Diffusion Model (RCDM) is a type of generative model that aims to map representations of data back to the original data space. It's a model that learns to generate data samples given their representations. Let's break down the components and concept of an RCDM:
1. Generative Models: Generative models are machine learning models that learn to generate new data samples that are similar to the training data they were exposed to. These models attempt to capture the underlying distribution of the data and use it to create new, realistic samples.
2. Diffusion Model: The term "diffusion" in this context refers to the process of gradually spreading or mixing information. In the context of generative models, diffusion models are used to model how a simple initial distribution (e.g., noise) transforms into a complex target distribution (e.g., real data) over time or through a sequence of transformations.
3.  Representation Conditional: The "representation conditional" aspect of an RCDM means that the model's generation process is conditioned on a given representation of data rather than random noise. This allows the model to generate samples that correspond to specific representations.
4.  An RCDM takes a learned representation (which could be a feature vector or embedding) of a data sample as input and aims to generate a data sample that corresponds to that representation. The key idea is to use a diffusion process to gradually transform a simple initial distribution (usually noise) into a sample that matches the given representation.
5.  Here, the researchers used RCDMs to map representations learned by SSL models back to the original image space. This allowed them to visualize how well the representations capture different aspects of the data, such as shape, pose, and color. By conditioning the diffusion process on the learned representation, RCDMs can generate samples that align with the information encoded in that representation.
6.  RCDMs have various applications, including image inpainting, style transfer, and data generation. They can be used to explore and manipulate the latent space (the space of learned representations) of a model and gain insights into what each dimension of the representation captures in terms of data features.
7. Here CDMs were likely used to visually assess how well different representations learned by SSL models preserved important features of the data, allowing researchers to better understand the strengths and limitations of these representations.

###### Experiment details
They trained several RCDMs on face-blurred versions of the ImageNet dataset, using the representations obtained from pretrained models with varying backbone representation sizes. They used the representations of two images—one from the ImageNet validation set and the other from a different source—as conditions for the RCDM. The RCDM generated multiple image samples using these conditions. Consistent aspects across these samples indicated information retained in the representation, while varying aspects represented information not contained in the representation

1.Training RCDMs: The researchers trained several Representation Conditional Diffusion Models (RCDMs). These models are designed to generate data samples based on given representations. In this case, the researchers trained these models using representations obtained from SSL pretrained models. These representations are typically high-dimensional vectors that capture important features of the original data, such as images.
2. Using Blurred Images: Instead of using the original ImageNet images, the researchers used versions of the images that were intentionally blurred. This introduces some level of distortion to the images while still maintaining their overall structure.
3. Generating Images: For each RCDM they trained, the researchers took two specific images—one from the ImageNet validation set and another from a different source. These images served as conditions for the RCDM. In other words, the RCDM uses these image representations as input and tries to generate images that correspond to them.
4. Understanding Retained and Varying Information: The RCDM generated multiple image samples using these conditions. The researchers then analyzed these generated samples. They looked for aspects that remained consistent across the generated samples. These consistent aspects indicated information that was retained in the original representation. For example, if the shape, pose, or color palette of the images remained consistent across the generated samples, it means these aspects were well-preserved in the representation.
5. Identifying Missing Information: On the other hand, the researchers also looked for aspects that varied across the generated samples. These varying aspects represented information that was not contained in the original representation. If certain features, details, or variations were inconsistent across the generated samples, it meant that the representation might not have captured these aspects well.


###### Observations
1. For SimCLR with a smaller backbone dimension (D = 512), the representation lacked information about shadows or vertical flips. However, using the largest backbone representation led to more accurate reconstructions, preserving pose, shape, and color.
2. Even in the case of images from a different source (out of distribution), the wider representation produced better-preserved information.
3. Comparing the supervised model's representation with smaller backbone dimension (D = 512) showed that it exhibited invariances in generated samples. This indicates that having a smaller D might lead to learning more invariances, potentially affecting robustness in various settings.

###### Verification of Hypothesis
1. An experiment on ImageNet-9 (subset of ImageNet) supported the hypothesis that a representation smaller than the number of classes is more robust to variations in the background.
2. However, while better performance and robustness were observed on ImageNet, these improvements did not necessarily carry over to other downstream tasks.

In summary, the qualitative visualization experiment using RCDM provided evidence that wider backbone representation sizes in SSL can better retain important information and reduce invariances. This insight offers a potential explanation for the improved performance observed in the experiments and highlights the complexity of the relationship between backbone dimensions, task-specific invariances, and downstream task performance.

## Number of parameters
In this section, the researchers explored the relationship between the number of parameters in the model and its performance, specifically focusing on how various ResNet variants behave in the context of self-supervised learning (SSL).

###### Number of Parameters and Model Performance:
1. Increasing the number of feature maps in the last convolutional block of a ResNet leads to a substantial increase in the total number of parameters in the network.
2. However, there's a question about how different ResNet architectures, such as those that are deeper or wider, fit into this framework and how they perform in SSL tasks.

###### Experiment and Results:
1. The results showed that many ResNet variants that were specifically designed to enhance performance in supervised training do not perform as well in the context of SSL.
2. Interestingly, a ResNet50 model with a wider backbone representation outperformed even deeper and wider ResNet variants when using the VICReg SSL approach.

In summary, the researchers found that increasing the number of parameters doesn't always lead to improved performance in self-supervised learning, as certain ResNet architectures tailored for supervised training might not adapt well to SSL tasks. The study's findings emphasize the complexity of the relationship between model architecture, parameter count, and performance in different learning paradigms.

![Number of parameters](https://github.com/Varchita-Beena/SSLHavenCorner/blob/SSLIncoming/Images/parameters.png)

## Wider SSL representations are sparse
This section of the research investigates the impact of using wider representations in self-supervised learning (SSL) on the sparsity of learned representations. Sparsity refers to the proportion of zero activations in a representation. 

###### Experiments and Results:
1. The researchers aimed to validate the hypothesis that when using a larger ratio of original dimension (D) to projected dimension (K), the learned representation becomes more sparse, meaning that a significant portion of the representation's values are zeros.
2. They conducted experiments using the VICReg SSL method and the ImageNet validation set.
3. In the first experiment, they used a backbone with a lower dimension (D = 512) compared to the projector's dimension (K = 8192).
4. In the second experiment, they used a backbone with a higher dimension (D = 32768) compared to the projector's dimension (K = 8192).
5. The evaluation involved computing the number of activations that are equal to zero in the representation for each example in the validation set.

###### Observations
1. In the case where the backbone's dimension is lower than the projector's (D < K), the amount of activations equal to zero is very low. This indicates that most examples span the entire backbone vector, and the representation is not particularly sparse.
2. In contrast, when the backbone's dimension is larger than the projector's (D > K), the representation becomes highly sparse. Around 80% of the examples have half of their backbone representation's activations equal to zero.

###### Significance
1. Using wider backbone dimensions leads to the learning of more information about the data.
2. The learned representations become more sparse when the backbone's dimension exceeds the projector's dimension.

In summary, the study suggests that employing wider representations in SSL not only enhances the richness of learned information but also encourages the emergence of more sparse representations.

## Wider representation are more easily linearly separable
This part of the research focuses on evaluating the performance of self-supervised learning (SSL) models using both linear and nonlinear probing methods. It aims to understand the separability of features learned by SSL models with varying backbone dimensions.

###### Experiment and Results:
1. Many SSL studies use linear probing, which evaluates how well the learned representations can be separated linearly, as a way to assess the model's quality.
2. However, this evaluation method can be limited if the learned information is entangled and not easily separable using linear methods.
3. To overcome this limitation, the researchers emphasize the importance of comparing the model's performance using both linear and nonlinear probing methods when evaluating SSL models.
4. The experiment involves comparing the performances of a model (SimCLR in this case) using both linear and nonlinear probing, while varying the dimension of the backbone (D).

###### Observations
1. For SimCLR, there's a noticeable performance gap between linear and nonlinear probing methods when the backbone dimension (D) is small. This indicates that the features learned with smaller dimensions are not easily separable using linear methods.
2. However, as the backbone dimension (D) increases, the differences in performance between linear and nonlinear probing decrease significantly.

###### Implication
1. Features learned with wider representation vectors (higher D) are more linearly separable.
2. As the dimension of the backbone representation increases, the learned features become more distinguishable and easier to separate using linear methods.

In essence, this observation highlights that SSL models with wider representations tend to learn more disentangled and linearly separable features, which can lead to better performance in tasks that require linear separability.

## Wider representations are binarizable
In this part of the research, the focus is on examining whether wider representations, which are more sparse as explained earlier, are also more easily binarizable.

"Binarizable" refers to the property of a representation or data that can be effectively converted or approximated into a binary format. In the context of the research, it means transforming the continuous values in the representation into binary values, typically 0s and 1s. This is often done by setting a threshold: values above the threshold become 1, and values below the threshold become 0. The research is investigating whether wider representations, which have more sparse activations, can be more easily transformed into binary values while retaining meaningful information.

###### Experiment and Results:
1. The intuition behind this is that if the dimensions of the embedding are roughly symmetrical, centered, and independent, quantization (binarization) should collapse different images to the same quantized code with very low probability. This probability decreases exponentially with the embedding dimension.
2. The intuition being discussed here is related to how well a high-dimensional representation (embedding) can be transformed into a binary format (binarization) while preserving its meaningful information. If the dimensions of the embedding have certain characteristics – they are symmetrical (similar distribution on both sides of a central point), centered around a specific value, and independent (not strongly correlated) – then when we convert these dimensions into binary values, different images should result in different binary codes with a very low chance of two different images having the exact same binary code.
3. The research is suggesting that if these conditions are met, the probability of two different images being mapped to the same binary code becomes extremely small, and this probability decreases very rapidly as the number of dimensions (embedding dimension) increases. In other words, when the embedding has these characteristics, it's easier to convert it into binary form without losing too much information, because each binary code can correspond to a unique pattern in the embedding space.
4. To test this hypothesis, the researchers run an experiment using SimCLR models trained with different backbone representation sizes (2048 and 32768).
5. They compare the performance of these models on various downstream tasks using both continuous and binarized representations.
6. The binarization operation is simple: all non-zero elements in the representation are set to one.
   
###### Observations:
1. For models with a representation size of 2048, the performances decrease significantly when the representation is binarized. This suggests that binarization of representations with smaller dimensions leads to a loss of information, affecting performance on downstream tasks.
2. However, for models with wider representations (32768), the performances between binarized and continuous representations are extremely close. This implies that wider representations can be successfully binarized while maintaining performance.

###### Implication:
1. Wider representations, due to their sparsity, exhibit characteristics that make them more amenable to binarization without significant loss of information.
2. Binarization of wider representations can lead to representations that are memory-efficient, as binary values require less storage space compared to continuous values.

In summary, this experiment demonstrates that wider representations have the ability to be more easily binarized without compromising performance on downstream tasks, making them a potentially efficient option for memory-constrained scenarios.

## Summary
1. Pretraining Bias and Robustness: Many SSL models are trained using a pretext task that might not align perfectly with downstream tasks. This can introduce a pretraining bias, leading to suboptimal performance on real-world tasks. The paper investigates how the ratio of the original dimension (backbone representation) to the projected dimension (projector) affects the robustness of SSL models to this bias.
2. Impact of Dimension Ratio: The research demonstrates that having a larger original dimension (backbone) compared to the projected dimension (projector) can limit the impact of the SSL loss on the representation. This leads to improved performance on downstream tasks, especially when dealing with real-world data that differs from the SSL pretext task.
3. Experiments: The paper conducts various experiments using linear and nonlinear projectors, changing backbone dimensions, and evaluating different SSL methods. It shows that wider backbone dimensions lead to better linear probe performance and improved accuracy on downstream tasks.
4. Sparsity and Linear Separability: Wider representations are found to be more sparse, meaning that many dimensions of the representation are close to zero. Additionally, they are more easily linearly separable, which is beneficial for downstream classification tasks.
5. Binarizability: Wider representations are also shown to be more easily binarizable without significant loss of performance on downstream tasks. This can help reduce memory requirements for storing representations.
6. Visualization and Interpretability: The paper uses a Representation Conditional Diffusion Model (RCDM) to visualize how representations capture different aspects of images. It demonstrates that wider representations can retain more information and produce more faithful reconstructions.
7. Comparison with Supervised Models: The research highlights that SSL models with wider representations outperform traditional supervised models on various tasks, showing the effectiveness of the approach.
8. Limitations and Future Work: The authors acknowledge that their study primarily focused on ResNet architectures. While their insights are valuable for such architectures, the study of other models, particularly vision transformers, requires more technical considerations that were left for future research.






















