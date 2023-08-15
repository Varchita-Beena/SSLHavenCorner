# Contrastive Loss
Contrastive loss measures the similarity between positive pairs of samples while maximizing the dissimilarity between negative pairs. In recent years, pre-trained models that employ contrastive loss have shown impressive outcomes and have even attained leading positions in self-supervised representation learning. However, the precise workings and efficacy of this loss function are still being actively investigated. Various research endeavors are dedicated to exploring the theoretical, experimental, and geometrical dimensions of this loss, aiming to unveil its complexities.
</br>
## Paper Title : [Tian et al., 2020](https://arxiv.org/abs/2005.10243) - What Makes for Good Views for Contrastive Learning?

### Outline
1.  Introduction
2.  Understanding the importance of invariant features from a story 'Funes the    Memorious' by Jorge Luis Borges.</br>
3.  InfoMax principle.</br>
4.  InfoMin principle.</br>
5.  Types of information captured. (Reverse U shaped relationship between an estimate of mutual information and representation quality).</br>
6.  Best views are downstream task-dependent</br>
7.  Creating views</br>
8.  Understanding Proposition</br>

#### Introduction
SimCLR is one of the famous approaches using contrastive loss. It maps images to a lower dimensional space and that's our embedding vector or trained representations. We want representations such that two different crops of same image should be close as much as possible i.e. they should be attracted to each other and two crops from different images should be repel each other. So different parts of the same image get represented alike while of different images end up away from each other in the embedding space. Two different views can be crops (disjoint crops or one crop is subset of another crop), color channels, etc.</br>

This idea is not new, this goes back to 1992 with the paper titled 'Self-organizing neural network that discovers surfaces in random-dot stereograms' by Hinton and Becker. This is essentially SimCLR but without deep networks.</br>

So, to set the goal clear - the objective is to learn an embedding that pulls positive pairs together and pushes negatives apart, where positive pairs are two views from the same image and negatives pairs are two views from different images. The numerator in the loss function is all about trying to increase the similarity of two positive views or a positive pair. The denominator is all about forcing two views of different images or negative pairs to map to different points in embedding space, i.e., they are to be pushed apart.</br>

###### InfoNCE Loss
![InfoNCE Loss](https://github.com/Varchita-Beena/SSLHavenCorner/blob/SSLIncoming/Images/UCL_W1_EQ1.png)
InfoNCE loss was introduced in the paper with title Contrastive Predictive Coding.</br>

#### Funes the Memorious
Isn't the concept of memory, perception, and the intricacies of human consciousness fascinating? Jorge Luis Borges' short story "Funes the Memorious" delves into these ideas. It examines the consequences of possessing a flawless memory – whether it's a blessing or a burden. The character Funes acquires an extraordinary mental capacity, becoming "memorious." He recalls every detail of his life in vivid precision, even minute aspects like cloud formations on specific days. However, this overwhelming memory prevents him from grasping abstract concepts. He struggles with meaningful conversations, literature, and creative thinking. Funes' all-encompassing memory hinders his ability to form connections and extract meaning from his experiences..</br>

Imagine encountering a piece of paper that feels unfamiliar, disconnected from your previous encounters with paper. In "Funes the Memorious," Funes becomes perturbed by the fact that a "dog at three fourteen (seen from the side) should have the same name as the dog at three fifteen (seen from the front)." This illustrates the potential danger of an excess of information. Our minds construct identity representations that discard unnecessary details. These view-invariant representations are crucial, especially in multiview coding research. For instance, after seeing a small piece of paper, our minds can seamlessly connect it to a larger one. As previously discussed, contrastive multiview learning involves aligning views of the same image and separating different views of distinct images. This demands remembering essential details while discarding irrelevant ones for abstract representation. The pivotal question is: "Which viewing conditions should remain invariant?" For tasks like classifying images based on the time of day, time invariance is impractical. Adopting Funes' approach of remembering every specific viewing angle would hinder our ability to track moving objects like dogs in videos or images.</br>

Consequently, achieving abstract yet useful representations is essential for downstream tasks. These representations shouldn't be so invariant that they lose the ability to recognize specific objects like dogs, paper, but also shouldn't discard too much information that would be useful for those tasks. In contrastive learning, the primary aim is to capture shared information between two views, making the choice of views crucial. Views can encompass various aspects like crops, disjoint crops, subsets of a larger crop, overlapping crops, different color channels, temporal slices, etc., all from the same data point. Striking a balance among these viewpoints to provide precisely the necessary information without an excess or deficiency becomes the challenge.</br>

The paper's authors employ two strategies to address this:
1. Downstream Task Alignment: When we're aware of the downstream tasks, we can tailor the design of effective views accordingly.
2. Identifying the Sweet Spot: Through empirical evidence, they demonstrate that, there's a "sweet spot" in terms of downstream performance. This sweet spot indicates a balanced level of mutual information (MI) between views – not too excessive nor too insufficient.</br>

This paper uses labeled data to learn better views but still perform contrastive learning using only unlabeled data. Also, the results depend on architectures also but to make results comparable authors only change input views, keeping other settings same.</br>

#### InfoNCE loss
Let's delve into the equation introduced at the beginning for the InfoNCE loss. We can deduce that the positive images originate from a combined distribution across both views, denoted as p(v1, v2), while the negative pairs stem from the product of individual distributions, p(v1)p(v2). Thus, the primary objective of contrastive learning is to develop an estimator for the mutual information between v1 and v2. This estimator is designed to differentiate between samples from the empirical joint distribution p(v1)p(v2|v1) and those from the product of marginal distributions p(v1)p(v2).</br>

Is it shown that the InfoNCE loss is the lower bound on the information shared between the raw views I(v1;v2). This relationship can be understood through the inequality sequence: I(v1;v2) ≥ I(z1;z2) ≥ InfoNCE loss. Here, z1 and z2 are the embeddings derived from applying specific model operations on the raw data. The rationale for this inequality lies in the data processing inequality, which states that local physical operations on a signal cannot increase its information content. This concept holds true here too, as the application of convolutions on images doesn't result in increased information after each operation. Hence, we have I(v1;v2) ≥ I(z1; z2) ≥ log(K) - InfoNCE loss, with K representing the number of negative samples. This implies that by minimizing the InfoNCE loss, we simultaneously maximize the information content between the raw images. Moreover, a higher count of negative images compels the model to work harder to find an optimal solution, thereby making InfoNCE an effective lower bound.</br>


#### InfoMin Principle
This was InfoNCE loss or InfoMax principle - suggests that in learning representations, the aim is to gather as much information as possible about the input. However, the authors introduce the InfoMin principle as its counterpart. They show that for effective performance in downstream tasks, the right set of views should share only the essential information. They argue that maximizing information is beneficial only when it's relevant to the task. Going beyond that, it's better to create representations that discard unnecessary details, which can improve overall performance and make tasks easier.

More concretly authors have given some definitions: - 
These definitions and concepts are about characterizing how well encoders preserve and represent information, especially in the context of contrastive learning and downstream tasks. They help us understand what kind of encoded representations are ideal for effectively using the learned information in various tasks.

###### Sufficient Encoder:
An encoder, f1, is considered "sufficient" in the context of contrastive learning when the mutual information (a measure of shared information) between the two views v1 and v2 is exactly the same as the mutual information between the encoded representation f1(v1) of v1 and v2. This means that during the process of encoding v1, the information related to v2 is preserved perfectly. In simpler terms, the encoded version of v1 (f1(v1)) contains all the necessary information from v1 that is relevant for the contrastive learning task involving v2.

###### Minimal Sufficient Encoder:
A "minimal sufficient encoder" is a special type of sufficient encoder. Among all encoders that are sufficient (preserve enough information for the contrastive task), a minimal sufficient encoder is one that does not include any additional irrelevant information. In other words, it extracts only the necessary and relevant information from the input. This is particularly useful when the views are constructed in a way that only specific information matters, and unnecessary details can be discarded.

###### Optimal Representation of a Task:
This definition is related to using the learned representations for downstream tasks, such as classification or prediction. For a specific task (let's say, predicting a label y based on input data x), an "optimal representation" z* is the encoded form of x that's both minimal and sufficient regarding the task at hand. In other words, z* contains only the most essential information needed to make accurate predictions about y based on x.

#### Information captured.
Authors created different views by randomly cropping two patches from the same image, each with varying offsets. They increased the spatial gap between the patches. After contrastive training, they evaluated the dataset by freezing the encoder and training a linear classifier. The mutual information vs. accuracy plot took a reverse-U shape. The reduction in InfoNCE loss led to an increase in mutual information, initially boosting downstream task accuracy. However, accuracy eventually declined when the patch distance was extremely high, resulting in minimal shared information. Similarly, with the smallest spatial distance, high shared information didn't yield optimal accuracy. Instead, a "sweet spot" emerged where the right amount of shared information was captured for the downstream task. The plot revolved around patch distances, and the paper showcased experiments with color space, color jittering, and random resized crops. In all cases, they observed a reverse U-shaped curve, indicative of this intriguing phenomenon.</br>

###### Missing information: 
We need to find a balance between how much information our views contain about the input and how effectively the learned representations work for a task. If we can't achieve the desired accuracy for a task, it means we're missing important task-related information. It's possible that the views are throwing away needed information, leading to poor performance.</br>

###### Excess information:
When we gather too much information between views, we end up with excess noise – unnecessary details that go beyond what's required. This includes noise or background information, which in turn reduces transfer learning accuracy and leads to poorer generalization. </br>

###### Sweet spot:
The sweet spot means getting just the right information, without any unnecessary noise – only the relevant details that matter for the task at hand.</br>
The analysis shows that transfer performance will have a limit (upper bound) represented by a curve that looks like a reverse U. The best performance, known as the sweet spot, is at the highest point of this curve.

Authors also show that the InfoMin principle can be put into practice by using more powerful data changes to lower mutual information, getting closer to the sweet spot. This approach led to achieving top-level accuracy (SOTA) on benchmark datasets. InfoMin Augmentation involves methods like RandomResizedCrop, Color Jittering, Gaussian Blur, RandAugment, Color Dropping, and a JigSaw branch. InfoMin pre-training consistently outperformed supervised pre-training as well as other unsupervised pre-training methods.

#### Best views are downstream task-dependent
Frequently, we lack a completely labeled training set that defines the downstream task beforehand. This makes it difficult to assess how much task-related information the views and representations contain during training. Constructing views usually involves using domain knowledge to modify the input while keeping the task-relevant factor intact.</br>

The auhtors build a toy dataset combines three tasks: </br>
1. Moving-MNIST: Videos of digits moving on a black canvas, bouncing off the edges.</br>
2. STL-10: A set background image.</br>
3. Colorful Moving MNIST: The final dataset with factors like digit class, digit position, and background class in each frame.</br>

In this setup:</br>
- View-1: A sequence of frames showing the moving digit.</br>
- View-2: A single image. Positive view shares a factor with the sequence, negative doesn't.</br>

For each image, three downstream tasks are considered:</br>
1. Predicting digit class.</br>
2. Localizing the digit.</br>
3. Classifying the background image (from STL-10's 10 classes).</br>
The authors freeze the main part of the model and train specific heads for each task.</br>

When only a single factor is shared between View-1 and View-2, the performance is significantly impacted. For instance, when only the background is shared, contrastive learning struggles to create representations that capture both digit class and location effectively.</br>

In scenarios where multiple factors are shared, one factor can overpower another. For instance, sharing the background might lead to latent representations ignoring important information about discriminating or locating digits. This could happen because background information dominates, causing the encoder to choose it as a shortcut to solve the contrastive pre-training task. However, when digit and position are shared, the digit-related aspects take precedence.</br>

#### Creating views
To experiment, the authors employ flow-based models to transform color spaces into new color spaces using InfoMin principle. These transformations create distinctive views by separating color channels. Contrastive learning and linear classifier evaluation are then performed on these views, all while preserving the properties of the color spaces during the transformation process. The experiments utilize pixel-wise operations and various types of flows, and the evaluation takes place on the STL-10 dataset.

For the above purpose, authors use adversarial training. They use "generator" that changes how pictures look. They also use "encoders," to figure out if the changed pictures are different or similar. The generator tries to make the changed pictures look similar, while the encoders try to tell them apart. This way, the generator learns to make pictures that are different but still look similar. The goal is to find a good balance between changing the pictures and keeping them similar. This helps the generator create useful and meaningful changes in the pictures. They use a specific method to measure the success of this process, and they make sure that the generator doesn't come up with meaningless or weird changes.

Experiments are done using different color representations like RGB and YDbDr. They notice the mutual information measure (INCE) and the accuracy of the tasks they're testing on follow a curve that looks like a reverse "U" shape. What's especially intriguing is that the YDbDr color representation is already close to the optimal point they are aiming for. This aligns with their belief that the way colors are separated into brightness and color information is a good strategy for capturing essential details while still making objects recognizable.

They also mention another color decomposition method called Lab, which performs similarly well. They point out that this method was designed to mimic how humans perceive color, which suggests that human color perception might actually be quite effective for self-supervised learning. 

However, they also note a challenge in their approach. The training process, which is similar to a type of machine learning called GAN, can be unstable. This means that different attempts with the same settings can lead to different results. They believe this instability might be because the view generator doesn't know anything about the final tasks they're testing on, so some constraints are not met perfectly.

To address this challenge, they propose a new method that combines both unsupervised and semi-supervised learning to improve stability and performance.

###### Semi-supervised view learning.
A method that leverages the availability of a small number of labeled examples for the downstream task. They want to guide the view generator, represented by "g," to retain the information about the labels. To achieve this, they introduce two classifiers, denoted as c1 and c2, which help in performing classification during the process of learning views. The goal is to optimize an equation that involves these classifiers and the mutual information measure (INCE). 

The INCE term applies to all data (both labeled and unlabeled), while the classifiers' terms are specific to the labeled data. In each iteration, they take both an unlabeled batch and a labeled batch and use the frozen view generator to create views for the unsupervised contrastive representation learning.

They show that, regardless of the original color space and whether the generator operates with volume-preserving (VP) or non-volume-preserving (NVP) flows (types of flow based methods), the learned views tend to be centered around the optimal performance region, or the "sweet spot." This outcome highlights the importance of incorporating information about the labels.

To further analyze their approach, they compare different types of view generators: "supervised," "unsupervised," and "semi-supervised". They also include the baseline of using contrastive learning over the original color space. The semi-supervised view generator significantly outperforms the strictly supervised one, which underscores the value of reducing mutual information between the learned views. 

They also compare the performance of their approach, with the raw input data using larger backbone networks. This comparison shows consistent improvement using the learned views over the raw input. This demonstrates the effectiveness of their method in enhancing the representation learning process.

#### Understanding Proposition
It states that by using minimal sufficient encoders, we can find optimal views that minimize mutual information between them while preserving relevant information for a downstream task. This process ensures that the views are well-suited for the task at hand.</br>

Given: Minimal sufficient encoders f1 and f2, and optimal views v1*, v2*.
Suppose we have minimal sufficient encoders, f1 and f2. For a downstream task T with label y, the optimal views (v1*, v2*) are chosen such that they minimize the mutual information between v1 and v2, under the constraint that the mutual information between each view and the label y, as well as the mutual information between the original data x and the label y, remain the same. They have given it's proof also.</br>
The proof establishes that when certain conditions are met (minimality and sufficiency of encoders), the optimal views derived from the data in the context of a downstream task lead to optimal representations through contrastive learning. </br>
The propositions and their proofs establish the relationships and properties of optimal views, minimal sufficient encoders, and optimal representations for downstream tasks in the context of self-supervised representation learning. </br>

The optimal views v1* and v2* for task T with label y are views such that:
I(v1*; v2*) = I(v1*; y) = I(v2*; y) = I(x,y)

###### Step 1: Starting Point and Mutual Information Equivalence:
Goal: To show that the optimal views for task T with label y satisfy the above mutual information conditions.

Given: Known equalities: I(v1;y) = I(v2;y) = I(x;y) 
Considering that v1 and v2 are functions of the data x, we want to connect these replationships to mutual information terms. 

###### Step 2: Applying Mutual Information Chain Rule:
Using chain rule I(a;b,c) = I(a;b) + I(a;c|b) for mutual information, we berak the mutula information I(y; x) into two parts involving views v1 and v2:
I(y;x) = I(y;v1, v2) = I(y;v1) + I(y;v2|v1)

###### Step 3: Utilizing Nonnegativity of Mutual Information:
The nonnegativity property of mutual information states that it is always greater than or equal to 0. I(a;b) ≥ 0
So I(y; v2|v1) ≥ 0

###### Step 4: Expressing Conditional Mutual Information:
Using the chain rule, we further express I(y;v2|v1) in terms of conditional mutual information involving x. We want to express I(y; v2|v1) in a way that helps us relate it to x and other conditional MI terms.
consider a = y, b = v1 and c= v2. We want tp express I(y; v2|v1)
I(y;v2|v1) = I(y; v2, v1) - I(y ; v1)
We can use the definition of v1 and v2 as functions of x and rearrange the terms:
I(y; v2|v2, x) = I(y; v2, x|v1) - I(y;x|v1)

##### Step 5: Relating Conditional Mutual Information:
Apply nonnegativity of MI and hence I(y; v2, x|v1) ≥ 0 and I(y; x|v1) ≥ 0

###### Step 6: Implications of Nonnegativity and Chain Rule:
Since all the terms in the expression for I(y; v2|v1) are non-negative, we can conclude that I(y; v2|v1) ≥ 0. But the problem statement asserts that I(y; v2|v1) = 0, this implies that both I(y; v2, x|v1) and I(y; x|v1) must be 0.

###### Step 7: Concluding Mutual Information Relations:
Now we can use the properties we've established to relate mutual information terms:
I(v1;v2) = I(v1;v2) + I(y; v2|v1) = I(v2; v1,y) = I(v2;y) + I(v2;v1|y) ≥ I(v2;y) = I(x;y)

This shows that I(v1*, v2*) = I(x;y) and that the optimal views (v1*, v2*) minimize the MI I(v1; v2) while satisfying the conditions

###### Step 8: Conditional Independence:
Finally v1* and v2* are conditionally independent given y because I(v2*; v1*|y) = 0 and this means that once the label is known, optimal views are independent of each other. 
This concludes the proof of Proposition 1, demonstrating that the optimal views for task T with label y have the desired mutual information relationships and conditional independence.

Given optimal views v1*, v2* and minimal sufficient encoders f1 and f2, the learned representations z1 or z2 are sufficient statistics of v1 or v2 for y, i.e., I(z1;y) = I(v1;y) or I(z2;y) = I(v2;y)

###### Step 1: Introduction and Goal:
The goal of this proposition is to show that the learned representations z1 (or z2) contain all the relevant information from v1 (or v2) for the label y, making them sufficient statistics. To do this, we start by breaking down the mutual information terms and using the properties of the encoders and optimal views.

###### Step 2: Breakdown of Mutual Information:
Break down I(y;v1) in terms of z1 and the relationship between v1 and z1: 
I(y;v1) = I(y; v1,z1) = I(y;z1) + I(y;v1|z1)

###### Step 3: Showing I(y;v1∣z1)=0:
To prove that I(y;v1)=I(y;z1), we need to show that I(y;v1∣z1)=0. Here's how this is done:
1. We can express I(y;v1∣z1) as I(y;v1)−I(y;v1;z1).
2. Using the chain rule of mutual information, we can expand I(y;v1;z1) as I(y;v1,v2)+I(y;v1∣v2)−I(y;v1;z1).
3. Continue to expand using mutual information properties, arriving at 
I(y;v1∣v2)+I(y;v1;v2∣z1)−I(y;v1;z1∣v2).
4. Utilize the property that optimal views v2 are conditionally independent given 
(the above proposition)  which implies that I(y;v1;v2∣z1)=0.
5. Use the nonnegativity property of mutual information to show that I(y;v1∣v2)≥0 and I(y;v1;z1∣v2)≥0.

###### Step 4: Establishing I(y;v1∣v2)=0:
By a similar reasoning process used in Proposition A.1, it can be shown that I(y;v1∣v2)=0. This step confirms that once we know v2, knowing v1 doesn't provide any additional information about y.

###### Step 5: Relating I(v1;v2∣z1)=0:
Now, the aim is to prove that I(v1;v2∣z1)=0, which will lead to I(y;v1∣z1)=0 and thus I(z1;y)=I(v1;y).
1. Recall that v1 and v2 are minimal sufficient encoders. According to Definition I(v1;v2)=I(v2;z1).
2. Using this and properties of mutual information, it follows that I(v1;v2∣z1)=0.

###### Step 6: Conclusion:
Now that I(v1;v2∣z1)=0, it implies that I(y;v1∣z1)≤0. By nonnegativity, we have  I(y;v1∣z1)=0. This concludes that z1 contains all the information from v1 relevant y, making it a sufficient statistic.

###### Step 7: Analogous Proof for The same logic can be applied to show that I(z2;y)=I(v2;y), which demonstrates that z2 contains all the information from  v2 relevant to y.

In summary, this proves that the learned representations z1 and z2 capture all the information necessary from v1 and v2 for the label y, confirming them as sufficient statistics. The proof uses properties of mutual information, minimal sufficient encoders, and the relationship between optimal views and encoders.

The representations z2 are also minimal for predicting y.
###### Step 1: Introduction and Goal:
The goal of this is to show that the learned representations z1 and z2 are not only sufficient statistics for predicting y, as shown in above, but they are also minimal sufficient statistics for predicting y.

###### Step 2: Establishing z1 as Sufficient Statistic:
Recall from Proposition A.2 that it has been shown that z1 is a sufficient statistic for predicting y, which means I(v1;y∣z1)=0. In other words, knowing z1 contains all the relevant information from v1 for predicting y.

###### Step 3: Analyzing I(z1;v1):
Now, the goal is to analyze I(z1;v1) and show that it is minimized, making z1 an optimal representation for predicting y.

1. Start with I(z1;v1)=I(z1;v1∣y)+I(z1;v1;y).
2. Use the fact that I(z1;v1∣y)=0 (as established in above), which means z1 already contains all the information from v1 needed to predict 
3. I(z1;v1;y) represents the shared information among v1, and y.

###### Step 4: Leveraging Sufficient Encoders:
For all sufficient encoders, it is proven that z1 is a sufficient statistic of v1 for predicting y. This means that any additional information from v1 given z1 is already contained in z1, making I(v1;y∣z1)=0.

Step 5: Simplifying I(z1;v1):
Using the properties established in previous steps, we can simplify the expression for I(z1;v1):
I(z1;v1)=I(z1;v1∣y)+I(v1;y)−I(v1;y∣z1)=I(v1;y)≥I(v1;y).

###### Step 6: Minimizing I(z1;v1) - Optimality:
In this step, the proposition argues that minimal sufficient encoders will minimize 
I(z1;v1) to I(v1;y), making z1 an optimal representation. This is because the goal of minimal sufficient encoders is to capture all the relevant information while reducing redundancy.

###### Step 7: Conclusion for z1:
By demonstrating that I(z1;v1)=I(v1;y) and I(v1;y∣z1)=0, the proof concludes that z1 is a minimal sufficient statistic for predicting y, which also makes it optimal.

###### Step 8: Similar Proof for z2:
A similar logical reasoning can be applied to show that z2 is also an optimal and minimal sufficient statistic for predicting y.

In summary, this proves that the learned representations z1 and z2 are not only sufficient statistics but also minimal sufficient statistics for predicting y. This is demonstrated by analyzing their information content, leveraging properties of sufficient encoders, and proving their optimality.


