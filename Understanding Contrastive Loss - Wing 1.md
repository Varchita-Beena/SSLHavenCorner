# Contrastive Loss
In recent years, pre-trained models that employ contrastive loss have shown impressive outcomes and have even attained leading positions in self-supervised representation learning. However, the precise workings and efficacy of this loss function are still being actively investigated. Various research endeavors are dedicated to exploring the theoretical, experimental, and geometrical dimensions of this loss, aiming to unveil its complexities.
</br>
## Paper Title : [Tian et al., 2020](https://arxiv.org/abs/2005.10243) - What Makes for Good Views for Contrastive Learning?
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








