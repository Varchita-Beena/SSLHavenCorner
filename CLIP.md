# [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)


## Outline
1. Introduction
2. Dataset
3. Natural language
4. Effifient pre-training method
5. CLIP
6. Few points
7. Pseudo code

## Introduction 
The authors of this study propose a straightforward and effective way to train state-of-the-art image representations from scratch using a large dataset of 400 million image-caption pairs gathered from the internet. The primary task during pre-training is to predict which caption corresponds to a given image. This approach is efficient and scalable, resulting in highly informative image representations.

After the pre-training phase, these learned representations are capable of associating natural language descriptions with visual concepts. This enables the model to perform zero-shot transfer to various downstream tasks without any task-specific fine-tuning. The study evaluates the performance of this approach on more than 30 different computer vision datasets, encompassing diverse tasks such as optical character recognition (OCR), action recognition in videos, geolocation, and fine-grained object classification.

Remarkably, the model's representations transfer effectively to many of these tasks. In many cases, the model's performance rivals that of fully supervised models that were trained with specific datasets for the respective tasks. For example, the model achieves the same level of accuracy as the original ResNet-50 model on ImageNet, even though it was not trained on the 1.28 million training examples that the ResNet-50 model used.

In summary, this study demonstrates the power of using a large dataset of image-caption pairs for pre-training image representations and leveraging the associated natural language descriptions for zero-shot transfer to a wide range of computer vision tasks. This approach achieves competitive performance without the need for extensive task-specific training data.

## Dataset
The existing research in the field of computer vision has predominantly utilized three datasets: MS-COCO, Visual Genome, and YFCC100M. While MS-COCO and Visual Genome are valuable datasets with high-quality crowd-labeled data, they are relatively small in size, containing around 100,000 training images each. In comparison, other computer vision models have been trained on much larger datasets, even up to 3.5 billion Instagram photos.

YFCC100M, with 100 million photos, is an alternative option. However, the metadata associated with each image in this dataset is sparse and of varying quality. Many images have automatically generated filenames or contain descriptions related to camera settings rather than descriptive content. After filtering out images with non-descriptive metadata and focusing only on those with natural language titles and/or descriptions in English, the dataset's size reduces by a factor of six to around 15 million photos. This size is approximately comparable to the well-known ImageNet dataset.

The motivation behind using natural language supervision is the vast amount of data in this format available publicly on the internet. Since existing datasets don't fully capture this potential, relying solely on these datasets for evaluation would underestimate the capabilities of research in this direction. To address this limitation, the authors of the study created a new dataset consisting of 400 million pairs of images and associated text. They collect this data from various publicly available sources on the internet. In constructing this dataset, they search for image-text pairs where the text includes one of a predefined set of 500,000 queries. The resulting dataset, named WIT (WebImageText), is designed to encompass a wide range of visual concepts. The dataset is approximately similar in terms of total word count to the WebText dataset used to train the GPT-2 language model.

## Natural language
Learning from natural language offers several advantages over other training methods. Scaling up natural language supervision is much simpler compared to traditional crowd-sourced labeling for image classification.

Conventional Approach: In traditional machine learning, especially for tasks like image classification, datasets are often created by having human annotators assign specific labels to images. For instance, in a dataset of animal images, each image might be labeled with the name of the animal it contains. This labeling is usually done using a "gold label" approach, where multiple annotators might vote on the correct label, and the majority label is chosen as the final annotation.

1-of-N Majority Vote "Gold Label": The term "1-of-N" refers to the common practice of representing labels as binary vectors where only one element is "on" (1) and the others are "off" (0). For example, in an animal classification task, if we have labels for cat, dog, and bird, "cat" might be represented as [1, 0, 0]. The "gold label" refers to the final chosen label, usually determined by majority voting among annotators.

Rigid Labeling: The traditional approach requires precise and consistent labels, which are used to guide the learning process. This labeling process can be quite structured and specific, often requiring a predefined set of categories and a standardized format for annotations.

Learning from Natural Language: In the context of learning from natural language, the supervision doesn't have to follow the strict conventions of traditional labeling. Instead of having predefined categories and fixed binary labels, natural language can provide more flexible and contextual guidance. For example, instead of a predefined "cat" category, a natural language description like "a fluffy feline with pointy ears" can guide the learning process.

So, when it's said that "learning from natural language doesn't demand such rigid labeling," it means that the guidance for learning can come from the richness of human language, allowing for more nuanced and flexible descriptions, as opposed to adhering strictly to predefined categories and standardized formats used in traditional machine learning labeling.

## Efficient pre-training method
Initial Approach: The researchers started with an approach that aimed to simultaneously train both an image analysis model (CNN) and a language analysis model (transformer) together from scratch. They wanted these models to predict captions for images.

Challenges Faced: However, when they tried to implement this joint training method, they encountered difficulties, particularly in terms of making this process efficient as they scaled it up.

Comparison with VirTex: They found that their transformer model, which had 63 million parameters (a measure of complexity), took more computing resources (about twice as much) compared to their ResNet-50 image analysis model. This transformer model, despite being more complex, learned to identify ImageNet classes (standard image categories) at a pace that was three times slower than a simpler baseline approach.

Baseline Approach: The baseline approach was simpler: it predicted a bag-of-words representation (a simpler way of representing text) of the image captions. Even though this approach was less sophisticated than the transformer model, it managed to learn about ImageNet classes faster.

In summary, the initial attempt of jointly training a complex transformer model for text and a CNN for images to predict captions proved to be computationally intensive and less efficient in learning about ImageNet classes compared to a simpler approach that predicted a basic bag-of-words representation of the captions.

Both the initial approach and the alternative approach have a common feature: they aim to predict the precise words in the textual descriptions that accompany images. However, predicting exact words in this context is challenging due to the wide range of different descriptions, comments, and other related text that can be associated with images. COntrastive objectives try not to predict the exact content, they learn by understanding the differences between various data samples. This approach has been found to be effective in image representation learning. generative models designed to create images can indeed produce high-quality representations, but they require significantly more computational resources than contrastive models with similar performance.

So, instead of predicting the exact words of the text associated with images, they attempted to predict which text, as a whole, is matched with which image. To achieve this, they modified their baseline approach (which used a bag-of-words encoding) and replaced the predictive objective with a contrastive objective. By doing this, they observed a notable improvement in efficiency. Specifically, they achieved a 4x improvement in the speed of zero-shot transfer to ImageNet, a widely used dataset for image classification. 


## CLIP
CLIP is a model trained to determine which pairs of images and text actually belong together in a given batch of data. To achieve this, CLIP learns a shared space where both images and text are represented as embeddings, and it aims to bring the embeddings of matching pairs closer while pushing apart the embeddings of non-matching pairs.

1. Training Setup: CLIP takes a batch of N (image, text) pairs. The goal is to determine which of these pairs are actual matches.
2. Embedding Space: CLIP has two components: an image encoder and a text encoder. These encoders transform images and text into numerical representations (embeddings) in the same shared space.
3. Similarity Objective: CLIP maximizes the similarity (specifically, the cosine similarity) between the embeddings of actual matching pairs in the batch. At the same time, it minimizes the similarity between embeddings of pairs that are not supposed to match within the batch.
4. Loss Function: The model uses a symmetric cross-entropy loss to optimize these similarity scores. This loss function guides the model to ensure that similar pairs have higher similarity scores and dissimilar pairs have lower similarity scores.
5. Batch Construction: The pairs are constructed within the batch, considering all the possible combinations of images and text samples in the batch. This creates N × N possible pairings. The model aims to figure out which of these pairings are true matches.

This training approach, which involves optimizing the similarity scores between embeddings, is not new. It draws on concepts from deep metric learning (multi-class N-pair loss), contrastive representation learning (InfoNCE loss), and even its adaptation to the domain of medical imaging for contrastive (text, image) representation learning.

The idea behind this approach is to enable CLIP to understand the inherent relationships between images and text, even when they are not explicit. By learning to differentiate between actual matches and non-matches within the same batch, CLIP can create embeddings that capture the semantics shared between images and their corresponding text descriptions.

Due to the extensive size of our pre-training dataset, the concern of overfitting is minimized. 
1. Training Initialization: They train CLIP from the ground up, without using any initial weights for the image encoder from ImageNet or pre-trained weights for the text encoder.
2. Projection Technique: Unlike the approach in literature, they do not employ a non-linear projection between the representations and the contrastive embedding space. Instead, they only use a linear projection to map the representations from each encoder to the shared multi-modal embedding space. They observed no noticeable difference in training effectiveness between the two techniques. The absence of non-linear projections suggests that these could be closely linked with the specifics of self-supervised representation learning methods.

Projection Technique: When we talk about projection, we're referring to the process of transforming the original representations of images and text into a shared embedding space. This is done to make these different types of data (images and text) comparable and closer in this shared space.

Non-linear vs. Linear Projection: In many cases, a non-linear function (like a neural network layer with activation functions) is used to perform this transformation. This allows the system to capture complex relationships between the data. However, the authors mention that they opted for a simpler approach by using a linear projection. A linear projection is like applying a linear transformation to the data, which can rotate, scale, and shift it, but it can't capture intricate non-linear patterns.

Contrastive Embedding Space: The embedding space is where both image and text representations are transformed to and compared. This comparison is essential for the model to learn that matching images and text should be close in this space.

Link to Self-Supervised Learning: The authors speculate that the use of non-linear projections might be more crucial in other self-supervised representation learning methods, where the tasks and objectives could demand capturing more intricate relationships. In the context of CLIP, where they are mainly focused on matching images and text, a linear projection sufficed.

3. Text and Image Transformation:
Originally, there was a process where a single sentence was randomly picked from the text data. This selected sentence was then used as part of the learning process. However, in their approach, they decided to skip this step. In other words, they removed the part of their training process that involved choosing one sentence from the text.

Similarly, for the images, they used a simpler approach to augment (enhance) the data during training. They applied a basic operation called "random square crop" to the resized images. This means they randomly selected a rectangular section of the image, and then resized it to make it a square.

4. Temperature Parameter:
In the context of their training process, they use a softmax operation. The softmax operation takes a set of numbers (logits) and converts them into a probability distribution. The temperature parameter (τ) in this context controls the "spread" or "range" of these logits before they are converted into probabilities. A higher temperature makes the logits more spread out, while a lower temperature makes them more concentrated.

In their approach, instead of fixing this temperature parameter beforehand (which would add an additional hyper-parameter to their model), they decide to optimize it directly during the training process. To do this optimization, they use a "log-parameterized multiplicative scalar." This means they manipulate the temperature parameter in a logarithmic way, which helps in making the optimization process more stable and prevents the temperature from becoming an additional hyper-parameter that needs tuning. The temperature parameter (τ) is adjusted during training from epoch to epoch.

Image encoder - They tried ResNet-50 with some modifications and Vision Transformer (ViT)
Text encoder - Transformer.

## Few points
Researchers studied zero-shot transfer as a way of measuring the task learning capabilities of machine learning systems. 

Researchers talk about challenges faced when using CLIP for zero-shot transfer on standard image classification datasets. In many of these datasets, the labels provided for images are often just numeric identifiers corresponding to the class labels. The actual names or descriptions of the classes in human language are not directly included in the dataset or are not well-structured for natural language understanding.

Because CLIP relies on the connection between images and their corresponding textual descriptions, this lack of direct text-based labels can create issues. One significant problem is polysemy, which means a single word can have multiple meanings depending on the context. Without contextual information, CLIP's text encoder can't differentiate between these different meanings. For example, the word "crane" could refer to a construction crane or a bird. In some datasets, both of these meanings might be treated as separate classes, which complicates the task.

Another issue is that some class labels might be ambiguous without context. For instance, the term "boxer" could refer to a breed of dog or a type of athlete. Without contextual cues, it's challenging for the text encoder to accurately understand the intended meaning.

In essence, the text-based part of CLIP requires meaningful and contextually rich class labels to effectively perform zero-shot transfer, but many existing datasets lack such labels, making the task more complex.

The researchers encountered challenges related to the nature of the text paired with images in their pre-training dataset. In their dataset, the text accompanying images is usually more than just a single word; it's often a full sentence that describes the content of the image in some way.

To address this distribution gap and ensure that CLIP can effectively use the text to understand the images, the researchers devised a strategy. They found that using a prompt template like "A photo of a {label}." helps to indicate that the text refers to the content of the image. This approach improves the performance of CLIP over using just the label text. For example, by using this prompt, they saw a 1.3% improvement in accuracy on the ImageNet dataset.

Drawing an analogy to "prompt engineering" seen in GPT-3, the researchers observed that customizing the prompt text for each task can significantly enhance zero-shot performance. They provided a few examples to illustrate this idea:

1. For fine-grained image classification datasets like Oxford-IIIT Pets, specifying the category in the prompt helped. For instance, using "A photo of a {label}, a type of pet."
2. For datasets like Food101, specifying a type of food in the prompt improved results.
3. For FGVC Aircraft, mentioning a type of aircraft in the prompt was beneficial.
4. For Optical Character Recognition (OCR) datasets, enclosing the text or number to be recognized in quotes improved performance.
5. For satellite image classification datasets, they found success with prompts like "A satellite photo of a {label}."

In summary, the researchers used different customized prompt templates to enhance CLIP's understanding of the images and their accompanying text, improving zero-shot performance across various tasks.

In their experimentation to enhance the performance of CLIP, the researchers explored the technique of ensembling, combining predictions from multiple models, and in this case, it's done in the embedding space using text embeddings. This approach has been shown to significantly boost the model's accuracy, especially when used in conjunction with different context prompts for prompt engineering.
1. Ensembling: In this case, the researchers used different "context prompts," which are variations of the phrases used to describe the content of an image.
2. Embedding Space vs. Probability Space: When we use ensembling, each model generates its own predictions. Traditionally, these predictions are probabilities (like the chances of an image belonging to a certain class). But in this case, the researchers combined the models' predictions not in the probability space, but in the "embedding space." The embedding space is a multidimensional representation of data, and in this case, it's about the similarity between the images and their descriptions.
3. Averaging Text Embeddings: Instead of averaging probabilities, the researchers averaged something called "text embeddings." These embeddings are representations of the descriptions of the images. By averaging these embeddings from different models, they created a single set of averaged embeddings.
4. Benefits of Embedding Space Ensembling: The clever part is that by working in the embedding space, the researchers were able to create the averaged embeddings and store them, which means the computational effort for ensembling remains the same as using a single model.
5. Results: The researchers found that combining the predictions of multiple models (ensembling) using different context prompts significantly improved their model's performance. For instance, on the ImageNet dataset, using 80 different context prompts for ensembling led to a 3.5% increase in accuracy compared to just using a single context prompt.

1. Zero-Shot CLIP Performance: The text-image model, CLIP, shows competitive performance on datasets where there's a separation between training and testing data splits. In these cases, the model's zero-shot performance (meaning it wasn't specifically trained on these tasks) is comparable to a simpler supervised baseline method. This baseline uses a linear classifier on top of features extracted from a ResNet-50 model, which is a common architecture for image classification.
2. Supervised Baseline: The supervised baseline method is basically a linear classifier added on top of the features extracted from ResNet-50. However, on most of these datasets, the performance of this baseline is not as high as the current cutting-edge methods. The "state of the art" refers to the best-known performance achieved on these tasks.
3. Challenges and Future Improvement: Despite CLIP's competitive performance, there's still a long way to go to improve its task learning and transfer capabilities. The paper suggests that while increasing the scale of training data and model size has helped improve performance so far, there's an estimated need for a substantial 1000-fold increase in computing resources (compute) for CLIP to achieve the same high performance as the best methods available today.
4. Limitations: However, such a large increase in computing power is currently infeasible due to hardware limitations. This means that while CLIP has potential, it's limited by the practical constraints of the available technology.
5. Future Research: The conclusion is that further research is needed to make CLIP more computationally and data-efficient. In other words, scientists need to find ways to make CLIP perform better without requiring such massive increases in computing resources. This could involve finding more efficient algorithms, improving data usage, or other innovative approaches.
6. Weak Performance on Specific Tasks: The analysis in paper highlights that CLIP's zero-shot performance is not strong on various types of tasks. When compared to models specifically designed for those tasks, CLIP's performance is lacking in certain areas.
7. Fine-Grained Classification: CLIP struggles with tasks that require fine-grained classification, such as distinguishing between different car models, flower species, and aircraft variants. These tasks demand a high level of detail and specific knowledge that CLIP's general training may not fully capture.
8. Abstract and Systematic Tasks: Tasks that involve abstract and systematic understanding also pose a challenge for CLIP. For instance, it might find it difficult to accurately count the number of objects in an image, which requires a deeper understanding of the scene.
9. Novel Tasks: CLIP's performance can also be poor on tasks that are not covered by its pre-training dataset. For example, if asked to classify the distance to the nearest car in an image, CLIP might perform at a level similar to random guessing.
10. Overall Performance Level: There are still many tasks where CLIP's zero-shot performance is close to random chance. This suggests that CLIP has limitations in handling a wide range of tasks, especially those that demand specific expertise or context beyond its training data.
11. While CLIP is effective at recognizing and understanding many different types of images that it was not explicitly trained on, it has difficulty when encountering images that are significantly different from what it has seen before. These unfamiliar or out-of-distribution images pose a challenge for CLIP's generalization capabilities.
12. While CLIP is effective at learning representations for digitally rendered text, which is common in its training data, it struggles with recognizing handwritten digits, as demonstrated by its performance on MNIST. In fact, a very basic approach like logistic regression on the raw pixel values of MNIST images performs better than CLIP's zero-shot recognition in this case.
13. CLIP's pre-training dataset lacks images that resemble MNIST digits, which means that the model didn't learn to generalize well to this specific type of data. This points to a limitation in CLIP's ability to handle certain types of variation and highlights that training on a diverse dataset doesn't guarantee that a model will generalize to all kinds of data. In essence, CLIP attempts to bypass this issue by training on a wide range of data, but it doesn't entirely solve the problem of models struggling with unfamiliar or unexpected variations.
14. While CLIP can create zero-shot classifiers for various tasks, it's constrained to selecting from the concepts it has learned and can't generate entirely new outputs like image captions can. This restriction is significant when compared to approaches like image captioning, where the model can generate novel textual descriptions for images.
15. They explored the idea of using image captioning as an alternative to CLIP. However, they found that the image captioning approach was computationally less efficient than CLIP. Therefore, they propose an alternative solution: joint training of a contrastive (similar to CLIP) and a generative (similar to image captioning) objective. This could potentially combine the computational efficiency of CLIP with the flexibility of a captioning model.
16. Another option they suggest is performing a search during inference over various natural language explanations for a given image. This approach is similar to what was proposed in the "Learning with Latent Language", where the model generates explanations in a flexible way during inference.
17. CLIP is trained using image-text pairs from the internet. These pairs are not filtered or curated, which means that the models learn various biases present in the data.
18. Paper provides a detailed analysis of these biases in CLIP and discuss potential ways to address or mitigate them.
19. CLIP doesn't directly tackle the challenge of inefficient data utilization in deep learning. Instead of directly addressing this issue, CLIP takes a different approach by using an extensive amount of training data as a way of supervision. It's designed to work with an enormous number of training examples, allowing it to scale to train on hundreds of millions of instances. If you were to show each of the 12.8 billion training images to a human observer, one image per second, it would take 405 years to go through all of them.
20. Paper talks about combining CLIP with other techniques that focus specifically on improving the efficiency of data usage. In particular, two methods are mentioned: self-supervision (where labels are generated from the data itself) and self-training (a process where a model is iteratively improved using its own predictions). These methods have shown their capability to make more efficient use of data compared to traditional supervised learning approaches.

## Pseudo code
1. Inputs:
I[n, h, w, c]: A minibatch of aligned images, where n is the batch size, h and w are the height and width of the images, and c is the number of channels.</br>
T[n, l]: A minibatch of aligned texts, where n is the batch size, and l is the length of each text sequence.

2. feature Extraction:
I_f = image_encoder(I): This step involves passing the images through an image_encoder (which could be a ResNet or a Vision Transformer) to obtain feature representations for each image. The result is a matrix I_f of shape [n, d_i], where d_i is the dimensionality of the image feature representation.</br>
T_f = text_encoder(T): Similarly, the text sequences are processed by a text_encoder (which could be CBOW or a Text Transformer) to obtain feature representations for each text. The result is a matrix T_f of shape [n, d_t], where d_t is the dimensionality of the text feature representation.

3. Learned Projection:
I_e = l2_normalize(np.dot(I_f, W_i), axis=1): The image feature representations I_f are projected into a common embedding space using a learned projection matrix W_i. The result is I_e, a matrix of shape [n, d_e], where d_e is the dimensionality of the shared embedding space. The l2_normalize function ensures that the vectors are normalized.</br>
T_e = l2_normalize(np.dot(T_f, W_t), axis=1): Similarly, the text feature representations T_f are projected into the same embedding space using a learned projection matrix W_t. The result is T_e, a matrix of shape [n, d_e].</br>

4. Similarity Computation:
logits = np.dot(I_e, T_e.T) * np.exp(t): The cosine similarity between image and text embeddings is computed by taking the dot product of the normalized I_e and T_e matrices. The multiplication by np.exp(t) scales the similarity values using a learned temperature parameter t.

5. Loss Calculation:
labels = np.arange(n): A label array is created with values ranging from 0 to n-1.
loss_i = cross_entropy_loss(logits, labels, axis=0): Cross-entropy loss is calculated for each image in the batch, treating text embeddings as ground truth labels along the columns (axis 0).</br>
loss_t = cross_entropy_loss(logits, labels, axis=1): Cross-entropy loss is calculated for each text in the batch, treating image embeddings as ground truth labels along the rows (axis 1).</br>
loss = (loss_i + loss_t)/2: The final loss is computed as the average of the losses from image and text predictions, resulting in a symmetric loss function.</br>

Overall, pseudo code represents the core process of training a multimodal embedding model using a symmetric loss function that aims to bring the embeddings of images and texts closer in a common embedding space. The learned projections W_i and W_t, as well as the temperature parameter t, are crucial components that enable the alignment of these two modalities in a way that maximizes their similarity.


