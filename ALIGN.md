# Paper: (Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision)[https://arxiv.org/abs/2102.05918]

## Outline
1. Introduction
2. Pre-training on Noisy Image-Text Pair
3. Transferring to Image-Text Matching & Retrieval
4. Transferring to Visual Classification
5. Results
6. Conclusion


## Introduction
This paper addresses the challenge of obtaining high-quality, large-scale representations for vision and vision-language tasks without relying on expensive or expert-curated datasets.
1. Background: In the field of natural language processing (NLP) and computer vision, pre-trained representations are crucial. These representations serve as foundational knowledge for various tasks, from text understanding to image recognition.
2. Data Challenges: Representation learning in NLP has evolved to the point where models can be trained on raw text without human annotations. However, vision and vision-language representations often rely on carefully curated datasets, which can be expensive and time-consuming to create. For vision tasks, datasets like ImageNet and OpenImages with explicit class labels are commonly used. Vision-language tasks typically use datasets like Conceptual Captions, MSCOCO, or CLIP, which require substantial data collection and cleaning.
3. Leveraging Noisy Data: This paper introduces a novel approach by leveraging a large, noisy dataset containing over one billion pairs of image alt-text. Notably, this dataset was obtained without undergoing expensive filtering or post-processing. The dataset is derived from the Conceptual Captions dataset.
4. Learning Architecture: The authors propose a straightforward dual-encoder architecture. This architecture is designed to align the visual and language representations of the image and text pairs using a contrastive loss function. The model's objective is to learn a representation that brings similar image and text pairs closer together in the learned feature space.
5. Scale Compensating for Noise: Despite the inherent noise in the dataset, the authors demonstrate that the sheer scale of the corpus compensates for this noise. The use of such a large dataset results in state-of-the-art representations, even with a relatively simple learning scheme.
6. Performance: The learned visual representations show strong performance when applied to various classification tasks, such as ImageNet and VTAB. These representations also enable zero-shot image classification, where the model can classify images into classes it hasn't seen during training. Additionally, the representations set new state-of-the-art results on image-text retrieval benchmarks like Flickr30K and MSCOCO, even when compared to more complex cross-attention models.
7. Cross-Modality Search: The learned representations facilitate cross-modality search, allowing complex text and text + image queries. This means that users can search for images using natural language queries, and the model can find relevant images based on the query.</br>
In summary, this paper introduces an innovative approach to representation learning for vision and vision-language tasks. By leveraging a massive dataset of noisy image alt-text pairs, the authors demonstrate that scale can compensate for noise in the data, leading to state-of-the-art representations. These representations enable a wide range of applications, from image classification to image-text retrieval and cross-modality search.

## Dataset
The main objective is to scale up representation learning by using a significantly larger dataset than what was previously available.</br>
1. Dataset Source: The dataset used in this work is constructed following the methodology used to create the Conceptual Captions dataset, which was introduced in 2018. Conceptual Captions is a dataset that pairs images with alt-text descriptions in English.
2. Scaling by Relaxing Cleaning Steps: Unlike the original Conceptual Captions dataset, which underwent extensive cleaning and filtering to ensure data quality, the authors of this paper intentionally choose to relax most of the cleaning steps. This decision is made to prioritize scale over data quality. As a result, the dataset obtained is significantly larger but noisier.
3. Data Filtering - Image-based: Pornographic images are removed from the dataset, aligning with ethical and content guidelines. Additionally, images with dimensions where the shorter side is less than 200 pixels or images with an aspect ratio greater than 3 are filtered out. Images associated with more than 1000 alt-texts are also discarded. The dataset is further cleaned by removing duplicates or near-duplicates of test images in all downstream evaluation datasets to avoid training on test images used for performance evaluation in other tasks.
4. Data Filtering - Text-based: Alt-texts (image descriptions) that are shared by more than 10 images are excluded. These alt-texts are typically irrelevant to the content of the images and are considered noise. Alt-texts containing rare tokens, i.e., words or phrases that do not appear frequently in the 100 million most common unigrams and bigrams from the raw dataset, are also discarded. Alt-texts that are either too short (less than 3 words) or too long (more than 20 words) are removed from the dataset. This filtering process helps eliminate noisy or irrelevant text descriptions.
5. In summary, the authors construct a massive dataset of image-text pairs by following the methodology used to create the Conceptual Captions dataset but relaxing many of the cleaning and filtering steps. While this results in a significantly larger dataset, it also introduces more noise. The filtering process involves removing inappropriate content, irrelevant alt-texts, and noisy descriptions to create a dataset that is suitable for scaling up representation learning, even though it may not be as pristine as smaller, carefully curated datasets.

## Pre-training on Noisy Image-Text Pairs
1. Dual-Encoder Architecture: ALIGN uses a dual-encoder architecture, which consists of two separate encoders – one for images and one for text. These encoders are used to generate fixed-size embeddings (or representations) of images and text, respectively.
2. Choice of Encoders:
Image Encoder: The image encoder is based on the EfficientNet architecture with global pooling. It's worth noting that the final classification layer of the EfficientNet (the 1x1 convolutional layer) is not trained.</br>
Text Encoder: For the text encoder, BERT is used, and embeddings are generated using the [CLS] token. The vocabulary used is generated from the training dataset and consists of 100k wordpiece tokens.</br>
3. Training from Scratch: Both the image and text encoders are trained from scratch, meaning their weights are initialized randomly.
4. Normalized Softmax Loss: The model is trained using a normalized softmax loss, which is a common loss function in contrastive learning.
5. Image-to-Text Classification: This loss measures how well the model can correctly associate images with their corresponding text descriptions. It encourages the model to give a higher similarity score (cosine similarity) to the correct text description for a given image.
![Image-to-text]()
6. Text-to-Image Classification: This loss is similar to the image-to-text loss but focuses on how well the model can associate text descriptions with their corresponding images.
![Text-to-image]()
7. Temperature Scaling: The temperature variable (σ) is introduced to scale the logits (the output of the model before applying softmax). It is used in the softmax function to control the smoothness of the probability distribution. The temperature is crucial because both image and text embeddings are L2-normalized. It allows the model to learn the optimal temperature value during training, rather than manually specifying it.
8. L2-Normalized Embeddings: L2 normalization means that all embeddings have a fixed length (magnitude) of 1. This can sometimes result in very sharp or concentrated probability distributions when passed through a softmax function, especially if the embeddings are highly discriminative.
9. Temperature and Calibration: Temperature allows us to control the "sharpness" of the probability distributions. When embeddings are L2-normalized, setting an appropriate temperature helps in calibrating the model's confidence.
10. Effective Learning: If we use a too-high temperature, the model's predictions may become overly diffuse and less informative. On the other hand, if the temperature is too low, the model may become overly confident and converge prematurely to suboptimal solutions. Finding the right balance is crucial for effective learning.
11. Batch Processing: To make in-batch negatives (negative samples within the same batch) more effective, embeddings from all computing cores are concatenated to form a much larger batch. This helps improve the quality of negative samples for training.

## Transferring to Image-Text Matching & Retrieval
The authors evaluate the ALIGN models they've developed on various image-text matching and retrieval tasks. They use several benchmark datasets to assess the model's performance.</br>
###### Benchmark Datasets:
1. Flickr30K: It contains 31,000 images, each paired with five descriptive captions. The task involves retrieving the most relevant caption for a given image or vice versa.
2. MSCOCO (Microsoft Common Objects in Context): It consists of over 120,000 images, each with five captions. The dataset includes a diverse range of scenes and objects.
3. Crisscrossed Captions (CxC): CxC is an extension of the MSCOCO dataset. It includes additional human semantic similarity judgments for caption-caption, image-image, and image-caption pairs. This extended annotation allows for various retrieval tasks and semantic similarity evaluations, including image-to-text, text-to-image, text-to-text, and image-to-image retrieval. It also includes tasks like semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS).

###### Evaluation:
The authors evaluate their ALIGN models on these benchmark datasets for image-text matching and retrieval tasks. They assess the models' ability to retrieve relevant information across different modalities, such as finding the most relevant image for a given text or vice versa.

###### Fine-Tuning:
The authors mention fine-tuning ALIGN models on the MSCOCO dataset. This means that after the initial pre-training on the large dataset with noisy image-text pairs, the model is further trained or fine-tuned on the specific MSCOCO dataset to adapt it to the tasks at hand.

###### Direct Evaluation on CxC:
Since the training set of CxC is the same as the original MSCOCO dataset, the authors can directly evaluate the performance of their MSCOCO fine-tuned ALIGN model on the extended CxC dataset. This allows them to assess how well the model generalizes from MSCOCO to more complex tasks introduced in CxC.


## Transferring to Visual Classification
###### Zero-Shot Transfer to Visual Classification:
The authors apply zero-shot transfer learning of ALIGN to visual classification tasks. They use the ImageNet ILSVRC-2012 benchmark dataset as the primary testbed. They also evaluate ALIGN on variants of ImageNet, including: ImageNet-R (ImageNet-Rendition) - This dataset consists of non-natural images, such as art, cartoons, and sketches; ImageNet-A (ImageNet-Adversarial) - It contains more challenging images specifically designed to challenge machine learning models; ImageNet-V2 - This dataset is an updated version of ImageNet, designed to address some of the limitations of the original dataset.

###### Transfer of Image Encoder:
The authors transfer the image encoder component of ALIGN to downstream visual classification tasks. This means they use the learned visual representation from ALIGN to assist in these tasks. They evaluate ALIGN's image encoder on various datasets, including: ImageNet - They report results for two settings -> Training only the top classification layer with a frozen ALIGN image encoder and Fully fine-tuning the model. Also on several smaller fine-grained classification datasets, including Oxford Flowers-102, Oxford-IIIT Pets, Stanford Cars, and Food101.

###### Robustness Evaluation:
The authors assess the robustness of their model on the Visual Task Adaptation Benchmark (VTAB). VTAB consists of 19 diverse visual classification tasks, each with 1000 training samples. These tasks cover subgroups of natural, specialized, and structured image classification tasks.

## Results
###### Zero-shot Visual Classification
Authors explains how ALIGN can perform zero-shot visual classification by encoding classnames as text descriptions. The authors compare ALIGN's performance with CLIP, demonstrating ALIGN's effectiveness and robustness in classifying images into classes it has never seen before. They use a prompt ensembling method similar to CLIP to enhance classification accuracy, resulting in improved performance on ImageNet classification tasks.

###### Visual Classification w/ Image Encoder Only
The authors perform visual classification tasks on the ImageNet benchmark using the ALIGN model. In this process, they freeze the learned visual features and train only the classification head at first. Then, they fine-tune all layers of the model. The results of ALIGN are compared with previous methods on the ImageNet benchmark, including CLIP, BiT, ViT, Meta Pseudo Labels, and others. ALIGN achieves state-of-the-art results in terms of top-1 accuracy with frozen features. After fine-tuning, it outperforms BiT and ViT models, with its main competition being Meta Pseudo Labels. ALIGN also demonstrates efficiency, saving 44% of FLOPS compared to NoisyStudent and Meta Pseudo Labels, while using a smaller test resolution. VTAB Evaluation - ALIGN is evaluated on a variety of visual classification tasks, with results reported in terms of mean accuracy and standard deviation. It outperforms BiT-L with similar hyper-parameter selection. ALIGN is evaluated on smaller fine-grained classification benchmarks using a simple fine-tuning strategy. ALIGN's results are compared to BiT-L and SAM, and it achieves comparable results to the state of the art, demonstrating its effectiveness in fine-grained classification tasks. In summary, this section details the fine-tuning and evaluation processes for visual classification tasks using the ALIGN model. ALIGN achieves state-of-the-art results on ImageNet and VTAB benchmarks and demonstrates competitive performance on smaller fine-grained classification tasks.

###### Ablation study
This section presents an ablation study conducted to analyze the impact of different factors and hyperparameters on the performance of ALIGN models, primarily focusing on zero-shot retrieval on MSCOCO and K-Nearest-Neighbor (KNN) tasks on ImageNet. The study aims to identify key architectural choices and hyperparameters that affect model performance. The study investigates the performance of ALIGN models using different combinations of image and text backbones. For image encoders, EfficientNet architectures ranging from B1 to L2 are used, while text encoders range from BERT-Mini to BERT-Large.Additional fully-connected layers with linear activation are added on top of selected image encoders to match the output dimension of EfficientNet-B7 (640). A similar linear layer is added to all text encoders.
1. The results show that model quality improves with larger backbones for both image and text encoders. However, the ImageNet KNN metric starts saturating when using EfficientNet-BERT-Large and EfficientNet-B7 with BERT-Large.
2. Scaling up the image encoder's capacity is more crucial for vision tasks, even when using BERT-Mini as the text encoder. For image-text retrieval tasks, both image and text encoder capacities are equally important.
3. The study explores key architecture hyperparameters, including embedding dimensions, the number of random negatives in the batch, and the softmax temperature.
4. Results show that model performance improves with higher embedding dimensions, with L2 using 1376 dimensions, which scales with the larger EfficientNet backbone.
5. Using fewer in-batch negatives (50% and 25%) in the softmax loss degrades performance, indicating the importance of having more negatives for training.
6. The effect of the temperature parameter in the softmax loss is studied. While some hand-selected, fixed temperatures perform slightly better than the baseline model, the learnable temperature parameter (converged to about 1/64) is chosen as it performs competitively and simplifies learning.
7. The temperature typically decreases quickly to around 1.2 times the converged values in the first 100k steps and then slowly converges until the end of training.

###### Pre-training Datasets
In this section, the authors investigate the performance of ALIGN models when trained on different pre-training datasets of varying sizes. Two model configurations are considered: EfficientNet-B7 + BERT-base and EfficientNet-B3 + BERT-mini. These models are trained from scratch on three different datasets:
1. Full ALIGN Training Data: This dataset represents the complete ALIGN training data, which is relatively large.
2. 10% Randomly Sampled ALIGN Training Data: This dataset is a smaller subset of the full ALIGN training data, comprising only 10% of the original data.
3. Conceptual Captions (CC-3M): CC-3M is a considerably smaller dataset compared to ALIGN. It contains around 3 million images.

1. Dataset Scale Matters: Larger-scale training datasets are crucial for achieving better model performance. Models trained on the full ALIGN training data clearly outperform those trained on the smaller CC-3M dataset.
2. Effect of Model Size: The choice of model size also plays a role. For instance, on the CC-3M dataset, the larger EfficientNet-B7 + BERT-base model starts to overfit and performs worse than the smaller EfficientNet-B3 + BERT-mini model.
3. Scaling with Dataset Size: Interestingly, the smaller model (B3+BERT-mini) tends to saturate at 10% of the ALIGN data, indicating that it doesn't fully benefit from a larger dataset. In contrast, the larger model (B7+BERT-base) shows clear improvement as the dataset size increases.
4. In summary, these findings emphasize the importance of both dataset scale and model size when training models like ALIGN. Larger datasets enable the scaling of models and contribute to better performance. Additionally, the choice of model size should be considered in relation to the dataset size to optimize performance.

###### Analysis of Learned Embeddings
1. The authors provide an analysis of the embeddings learned by the ALIGN model and showcase the model's ability to perform image retrieval based on text queries and image+text queries.
2. The authors build an image retrieval system using an index of 160 million images that are separate from the ALIGN training set.
3. ALIGN demonstrates its capability to retrieve precise images when provided with detailed textual descriptions of scenes, landmarks, artworks, and other fine-grained or instance-level concepts.
4. This illustrate ALIGN's capacity to align images and texts with similar semantics and its ability to generalize to novel and complex concepts.
5. The authors observe linear relationships between image and text embeddings in ALIGN, akin to those seen in word embeddings like word2vec.
6. To explore this phenomenon, they perform image retrieval using a combined image+text query. The embeddings of the query image and text string are added together to retrieve relevant images.
7. This approach introduces a new paradigm of "search with multi-modal queries," enabling searches that would be challenging using solely text or image queries. For example, users can find equivalents of objects or attributes in different contexts or subtract objects/attributes from a scene through operations in the embedding space.
Overall, these findings demonstrate the versatility and semantic alignment capabilities of the ALIGN model, making it a valuable tool for various applications in computer vision and natural language understanding.

###### Multilingual ALIGN Model
1. The authors introduce a multilingual version of the ALIGN model, referred to as ALIGNmling.
2. ALIGNmling is trained on a multilingual dataset that extends the existing English dataset to cover over 100 languages.
3. The dataset consists of noisy web image text data, and it's noteworthy that none of the data filtering processes are language-specific.
4. To accommodate the multilingual aspect, a new multilingual wordpiece vocabulary with a size of 250k is created to represent words across all supported languages.
5. The training configuration and methodology for ALIGNmling follow that of the original ALIGN model trained in English.
6. The authors evaluate ALIGNmling on the Multi30k dataset, which is a multilingual extension of the Flickr30K dataset, including English, German, French, and Czech.
7. Multi30k comprises 31,783 images, each with five captions in English and German and one caption in French and Czech.
8. The evaluation metric used is mean Recall (mR), which calculates the average score of Recall@1, Recall@5, and Recall@10 for image-to-text and text-to-image retrieval tasks.
9. ALIGNmling outperforms M3P on all languages, achieving substantial improvements, with the most significant improvement being +57.8 absolute mR on French (fr).
10. ALIGNmling's zero-shot performance is even comparable to fine-tuned models (models trained on the same language as the evaluation data) like M3P and UC2, except for Czech (cs).
11. However, ALIGNmling performs slightly worse on English (en) compared to its English-only counterpart, ALIGNEN, which was trained solely on English data.
12. ALIGNmling demonstrates strong multilingual performance on the Multi30k dataset, outperforming existing models in zero-shot retrieval tasks across different languages. This highlights the model's versatility and its ability to handle diverse languages for image-text retrieval.

## Conclusion
1. This research presents a novel approach to scaling up visual and vision-language representation learning without the need for extensive data curation and annotation.
2. Data Collection and Training Methodology: The authors leverage a massive dataset of noisy image-text pairs, obtained without the need for intricate filtering or expert curation. This dataset is significantly larger than existing ones and is derived directly from raw alt-text data, avoiding the manual construction of allowlists or reliance on high-frequency visual concepts.
3. Model Architecture: They propose a straightforward dual-encoder architecture for training ALIGN. This model consists of image and text encoders, which are trained jointly using a contrastive loss function. The simplicity of this architecture is noteworthy, as it achieves impressive results.
4. Cross-Modal Retrieval: ALIGN exhibits strong performance in cross-modal retrieval tasks, surpassing state-of-the-art (SOTA) models in both visual-text and text-visual retrieval tasks. This indicates its ability to understand and connect information across different modalities effectively.
5. Visual-Only Downstream Tasks: ALIGN's capabilities extend beyond cross-modal retrieval. In visual-only downstream tasks, it either matches or outperforms SOTA models trained on large-scale labeled data. This highlights its versatility and effectiveness in various computer vision applications.
6. Comparison with CLIP: A related model, CLIP, is compared to ALIGN. The key difference lies in the training data: ALIGN is trained on raw alt-text data without the need for an allowlist, while CLIP constructs a dataset based on high-frequency visual concepts from English Wikipedia. Despite this difference, ALIGN achieves strong performance, underscoring the significance of its training methodology.
7. An "allowlist," also known as a "whitelist," is a list of items or entities that are explicitly permitted, accepted, or approved in a particular context. In the context of data collection and curation, an allowlist is a pre-defined list of specific items or concepts that are considered acceptable or relevant for inclusion in a dataset.
8. CLIP's data collection process involved constructing an allowlist of high-frequency visual concepts from English Wikipedia. This means that CLIP's dataset was created by selecting and including only those visual concepts that were on the allowlist.
9. ALIGN, on the other hand, did not rely on such an allowlist. It was trained on raw alt-text data, which means it used a much larger and less restricted dataset of image-text pairs without the need for predefining a list of acceptable visual concepts.

Overall, this work demonstrates that with a large and noisy dataset and a straightforward training approach, it is possible to obtain robust visual and vision-language representations. ALIGN's impressive results in various tasks underscore its potential for a wide range of applications, from cross-modal retrieval to visual-only tasks. Moreover, it highlights the feasibility of scaling up representation learning without the need for extensive data curation, making it a valuable contribution to the field of artificial intelligence and machine learning.



