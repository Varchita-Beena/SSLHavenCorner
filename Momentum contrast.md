# Momentum Contrast for Unsupervised Visual Representation Learning
## Outline
1. Introduction
2. MoCo

## Introduction
Natural langugae processing: The data involves discrete elements like words or sib-word units that can be organized into tokeinzed dictionaries. This enables the application of unsupervised learning methods, where the goal is to learn meaningful representations from the data. These representations capture the relationships and meanings of the words or tokens within the language. 

For computer vision the data is quite different, images are continuous and exist in a high-dimensional space, making it challenging to directly apply the same strategies used for language tasks. Images are not naturally structured in a way that's easily interpretable for human communication, unlike words in a language.

Nonetheless, recent studies have shown promising results in unsupervised visual representation learning using methods similar to the contrastive loss. Despite the differences between language and computer vision data, these methods can be thought of as creating dynamic dictionaries for images. Here's how:
1. Dynamic Dictionary Building: Instead of fixed dictionaries as seen in language tasks, dynamic dictionaries are created for images. The "keys" in this dictionary correspond to sampled data points (such as image patches), and each key is represented by an encoder network. These keys capture specific patterns or features within the data.
2. Learning Process: Unsupervised learning is employed to train these encoder networks. The goal is to make sure that when a new data point (a "query") is encoded, it should be similar to the key that matches its underlying pattern, while being dissimilar to other keys that represent different patterns.
3. Contrastive Loss: The learning process is framed as minimizing a contrastive loss. This loss function encourages the encoded queries to be close to their corresponding keys while being separated from other keys. In essence, the network learns to distinguish between different visual patterns or features present in the images.

So, despite the challenges posed by the continuous and high-dimensional nature of image data, these methods leverage the concept of dynamic dictionaries and the idea of matching queries with keys to develop meaningful visual representations. This enables computers to understand and interpret images in a way similar to how they process language.

## Method
### Contrastive Learning as Dictionary Look-up
In contrastive learning, the underlying idea is to treat the learning process as if we're building a dictionary where encoded queries are matched to specific keys. The goal is to make sure that the encoded queries are similar to their corresponding positive keys (those that match the query) and dissimilar to all other keys (considered negative keys).

Here's how this process works:

1. Encoded Queries and Keys: Consider that we have an encoded query q and a set of encoded samples {k0, k1, k2, ...}, which serve as the keys in the dictionary. Each encoded sample represents a specific pattern or feature within the data.

2. Positive Key: Among all the keys, there's a single key (denoted as k+) that matches the query q. This is the "positive" key that we want the query to be similar to.

3. Contrastive Loss Function: The contrastive loss is a function that quantifies how well the query matches its positive key compared to the other keys. It aims to have a low value when the query is similar to its positive key and dissimilar to the rest of the keys. One common form of contrastive loss is InfoNCE (Normalized Cross-Entropy), which is expressed as a softmax-based classification problem.

4. Training the Encoder Networks: The contrastive loss serves as an unsupervised objective function for training the encoder networks. These networks are responsible for encoding the queries and keys. The input samples can be image patches or context that includes a set of patches.

5. Instantiation of Networks: The specific design of query encoder and key encoder can vary based on the particular pretext task at hand. These networks can be the same, partially shared, or completely different, depending on the requirements of the learning problem.

In summary, contrastive learning treats the learning process as constructing a dictionary where queries are matched to specific keys. The contrastive loss ensures that the encoded queries are encouraged to be similar to their corresponding positive keys while being distinct from other keys, leading to the learning of meaningful representations from the data.

### MoCo
This "dictionary" is unique in that it's dynamic – its keys are chosen randomly, and the way these keys are transformed (encoded) evolves as the learning process progresses. The core idea behind this approach is that valuable features can be acquired by using a comprehensive dictionary that includes a diverse set of negative examples. Meanwhile, the encoding process for the dictionary's keys should remain as consistent as possible, even though it is changing over time due to training. Momentum Contrast is a technique that builds upon the concept of contrastive learning to learn effective features from data

###### Dictionary as a queue.
The concept of the "dictionary" is realized as a queue that holds data samples. Imagine we're trying to learn about different types of animals by looking at pictures. We want to build a collection of important features that help us tell apart these animals. But there's a challenge: we have a lot of animal pictures, and we can only study a few at a time. Then here the queue-based dictionary approach helps:

1. Reusing What You've Learned: As we study one set of animal pictures, we jot down important features that help us recognize them. Instead of throwing away this information after we're done, we keep it handy, like having a cheat sheet.
2. Separating Collection Size: The size of our collection of features (our "dictionary") doesn't have to match how many pictures we're looking at each time. We can have a big collection of features even if we're only looking at a few pictures at once.
3. Keeping Things Fresh: As we look at more pictures, we want to update our collection to include the new things we've learned. But we don't need to keep everything forever. We can replace the oldest information with the newest stuff to keep our collection relevant.
4. Managing Workload: Even though it takes some extra effort to keep track of all the information, it's worth it because we're able to learn more efficiently. Removing the oldest information actually helps us stay focused on what's most important now.

So, the "dictionary as a queue" is like having a growing and changing set of notes that we keep using to learn from new pictures. This approach helps us build a strong understanding of the animals without getting overwhelmed by all the pictures at once.

Linking this concept to MoCo:
1. Reuse of Encoded Keys: By maintaining the dictionary as a queue, we can leverage the encoded keys from the immediately previous mini-batches. This means that the encoded keys aren't discarded after one use; instead, they are kept in the queue and can be used again. This reusability of keys is made possible due to the queue structure.
2. Decoupling Dictionary and Mini-Batch Sizes: The size of the dictionary is not restricted by the size of the mini-batch. In other words, we can have a much larger dictionary compared to the size of a single mini-batch. This flexibility allows us to set the dictionary size as a separate hyper-parameter, enabling more effective representation learning.
3. Dynamic Replacement of Samples: The samples within the dictionary are not static; they are progressively replaced. With each new mini-batch, the current batch of samples is added to the queue, and the oldest batch in the queue is removed. This dynamic replacement ensures that the dictionary always represents a sampled subset of all the available data.
4. Managing Computational Load: Despite the additional computational load of maintaining this dictionary, it remains manageable. The extra effort required to keep track of the queue is balanced by the benefits it provides for reuse and dynamic updating of encoded keys.
5. Removing Outdated Information: The process of removing the oldest mini-batch from the queue can be advantageous. Since the oldest batch's encoded keys are the least consistent with the newest ones due to the evolving nature of the network, removing them helps maintain the overall quality and relevance of the dictionary.

In summary, the dictionary, in the form of a queue, allows us to efficiently manage encoded keys, make use of past mini-batches, and keep the representation learning process flexible and adaptable to different data sizes.

###### Momentum update.
Imagine we have this "queue" of data samples that we're using to learn important features from, like we discussed earlier with the animal pictures. Now, we not only want to learn from these samples, but we also want to improve how we're learning, using "back-propagation" which adjusts how we learn based on the mistakes you make.

However, using a queue can be a bit tricky. If we just directly copy the way we're learning from new data to the old data in the queue, we might not get the best results. This is because the queue contains a mix of old and new information, and they don't play well together when we apply the same learning strategy to both.
So, here's where the "momentum update" comes in:

1. Maintaining Consistency: Think of our learning process as a recipe for making a cake. We have a mix of old and new recipes, and we want to bake a cake that tastes good using all of them. But sometimes, if we mix them all together at once, it might not taste great. The "momentum update" is like adding the ingredients from the old recipes gradually, so they blend well with the new ones, creating a cake that's both new and still familiar.

2. Gradual Change: Imagine we're painting a picture and you want to change the colors we're using smoothly. If we suddenly switch from red to blue, it might look weird. But if we mix a little blue with the red, we'll get a nice transition. Similarly, the "momentum update" helps our learning process transition from the old way to the new way gently, avoiding sudden changes that could mess things up.

3. Momentum Coefficient: Think of the momentum coefficient as the strength of the blending. If we set it high, like 0.999, it means we're giving more importance to the old way of doing things. It's like if we're learning to ride a bike, and we really want to remember the old tricks our friend taught you. If we set it lower, like 0.9, it means we're paying more attention to the new way. It's like if we're learning a new dance move and we're excited to try out the new steps. In their experiments, they found that letting the old way have more influence works better because it keeps things steady and familiar.

So, in simple terms, the momentum update helps combine old and new learning in a way that feels right, by gently blending them together and deciding how much weight to give to each.

Linking this concept to MoCo:
1. Maintaining Consistency: The momentum update is a way to help the old information in the queue stay relevant while still allowing the new information to shine. It's like blending the old and new ways of learning so that they work together better.
2. Gradual Change: The momentum update smoothly combines the learning approaches for old and new data. This way, even though the queue contains data from different times, the learning method doesn't change too abruptly.
3. Momentum Coefficient: The "m" in the formula is the momentum coefficient. It's like a knob we can adjust. If you set it higher (like 0.999), the old information has more influence, and if you set it lower (like 0.9), the new information has more influence. In their experiments, they found that a higher momentum value works better, suggesting that a slow and consistent update for the old information is key.

So, the momentum update is like a strategy that helps us blend what we've learned from different times in a smooth and effective way, making sure we benefit from both old and new data without causing confusion.

### Relations to previous mechanisms.
###### End-to-End Update by Back-Propagation:
Analogy:
Imagine we're in a cooking class with our friends. We're all learning to make a delicious dish using the same recipe. Each friend follows the recipe, and because we're all using the same recipe, the dishes we create are quite similar – they all taste like the same dish.

Now, the classroom we're in is like our computer's memory or GPU. It's where we have all our cooking tools and ingredients. Just like the classroom has limited space, our computer's memory has a limit too, which means we can only work with a certain number of "students" (mini-batch size) at a time.

If we want to involve more friends in the cooking class, we need a bigger classroom. Similarly, if we want to use a larger mini-batch size, we need a computer with more memory. But there's a challenge – bigger classrooms or computers can sometimes be harder to manage or might not be available.

Connection to the Mechanism:
In the analogy, the cooking class with the same recipe for everyone is like the "end-to-end update by back-propagation" mechanism. Each friend's dish (encoded representation) is consistent with others because they all followed the same recipe.

The limitation of this mechanism is similar to how the classroom size (GPU memory) affects how many friends (mini-batch size) we can have. If we want to involve more friends (increase mini-batch size), we need a bigger classroom (more memory), but that might not always be practical.

So, in this mechanism, the consistency of dishes (encoded representations) is high, but we're limited by the classroom size (memory) for accommodating more friends (samples) in our cooking class (training process).

###### Pretext Tasks Driven by Local Positions:
Analogy:
Imagine we are playing a game with cards. The goal of the game is to place the cards on a table following certain rules. The position where we place each card is like a local position in our learning process. We play this game several times, each time trying out different positions to see what works best.

Now, because we're playing multiple rounds of the game, we can use more positions on the table – this allows us to explore different ways to place the cards. Having more positions means we have a bigger "playing area" (dictionary size) to work with.

Connection to the Mechanism:
In the analogy, the game we're playing with cards is like the "pretext tasks driven by local positions" mechanism. Each round of the game corresponds to a different local position in the learning process. Using various positions allows you to learn more variations and potentially have a larger dictionary (more diverse representations).

However, there's a challenge – each round of the game might have its own special rules or tools (network designs like patchifying or changing receptive fields) that we need to follow. This can make things more complicated, just like how using special rules might make it harder to apply the skills you learned in other situations.

So, this mechanism allows for a larger dictionary size by using multiple positions, but it comes with the challenge of needing specialized rules or tools, which might not easily translate to other tasks or scenarios.

###### MoCo's Approach:
Now, let's say we have a recipe book and a kitchen, just like the previous cooking class, but this time we have a helper who can remember past recipes and help us mix and match. This helper is like the "momentum update" in MoCo. It's taking some ideas from the old recipes and blending them with the new ones, making sure the taste stays good. This way, we're not limited by the size of the kitchen or the need for special rules. We can try new things while keeping a touch of familiarity.

The "momentum update," allows for a larger dictionary without complicated rules and special designs, while still keeping things consistent and familiar.

###### Comparison between the "memory bank approach" and the "MoCo" approach




















