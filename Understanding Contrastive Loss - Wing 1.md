# Contrastive Loss
In recent years, pre-trained models that employ contrastive loss have shown impressive outcomes and have even attained leading positions in self-supervised representation learning. However, the precise workings and efficacy of this loss function are still being actively investigated. Various research endeavors are dedicated to exploring the theoretical, experimental, and geometrical dimensions of this loss, aiming to unveil its complexities.
</br>
## Paper Title : [Tian et al., 2020](https://arxiv.org/abs/2005.10243) - What Makes for Good Views for Contrastive Learning?
SimCLR is one of the famous approaches using contrastive loss. It maps images to a lower dimensional space and that's our embedding vector or trained representations. We want representations such that two different crops of same image should be close as much as possible i.e. they should be attracted to each other and two crops from different images should be repel each other. So different parts of the same image get represented alike while of different images end up away from each other in the embedding space. Two different views can be crops (disjoint crops or one crop is subset of another crop), color channels, etc.</br>

This idea is not new, this goes back to 1992 with the paper titled 'Self-organizing neural network that discovers syrfaces in random-dot stereograms' by Hinton and Becker. This is essentially SimCLR but without deep networks.</br>

So, to set the goal clear - the objective is to learn an embedding that pulls positive pairs together and pushes negatives apart, where positive pairs are two views from the same image and negatives pairs are two views from different images. The numerator in the loss function is all about trying to increase the similarity of two positive views or a positive pair. The denominator is all about forcing two views of different images or negative pairs to map to different points in embedding space, i.e., they are to be pushed apart.</br>

###### InfoNCE Loss
![InfoNCE Loss](https://github.com/Varchita-Beena/SSLHavenCorner/blob/SSLIncoming/Images/UCL_W1_EQ1.png)
InfoNCE loss was introduced in the paper with title Contrastive Predictive Coding. 





