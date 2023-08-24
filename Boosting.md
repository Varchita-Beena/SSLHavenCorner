# Boosting

## Outline
1. Intriduction
2. PAC Learning in brief
3. PAC Learning with an example
4. Fundamental Theorem of Learning Theory
5. Weak Learnability

Imagine we have a bunch of little animals, and we want to make them work together to do something smart. But some animals are not very good at figuring things out, they're a bit weak. 

So, what we can do is teach these weak animals a little bit about what's right and wrong. We give them some simple rules to follow. These rules might not be perfect, but they're better than guessing randomly.

Then, we ask all these animals what they think about a question. They might not be very smart alone, but when you put all their answers together, it's like they become one smart animal!

And guess what? This group of animals is called a "Boosting" team. They help each other to solve problems that none of them could solve alone. They learn from each other's mistakes and make a strong decision together.

So, just like how friends can help you solve puzzles better than we can alone, these animals (or models) work together to be smarter and make better decisions! That's how boosting works in a simple way.

Born out of a theoretical question and has proven to be a valuable tool in machine learning - Concept of boosting. Addresses two major challenges in machine learning.

1. bias-complexity tradeoff: There's a tradeoff between the complexity of the model and its bias towards the training data. More complex models can fit the training data very closely, but they might also overfit and not generalize well to new data. On the other hand, simpler models might not capture the underlying patterns accurately. Boosting tackles this challenge by starting with a basic model, which might have some bias and approximation errors, and then progressively improving it by adding more complex models. This way, the learning process smoothly controls the balance between approximation error and estimation error.

A "basic model with some bias" refers to a simple and straightforward predictive model that might not capture all the complexities of the underlying data distribution. This basic model, often referred to as a "weak learner," could be less accurate on its own due to its inherent limitations, biases, and simplistic nature.

2. Computational complexity of learning: For many complex concept classes, finding the best hypothesis (model) that minimizes the error might be computationally very expensive or even infeasible. Boosting comes to the rescue by enhancing the accuracy of weak learners. These weak learners are simple models that perform slightly better than random guessing. By efficiently aggregating these weak hypotheses, boosting creates increasingly accurate predictors for larger and more complex classes, even if the individual weak learners are relatively simple.

The authors focus on a specific boosting algorithm called AdaBoost, short for Adaptive Boosting. AdaBoost works by combining multiple simple hypotheses to create a final hypothesis. This final hypothesis is essentially a linear combination of these simple hypotheses. The family of hypothesis classes that AdaBoost employs involves stacking a linear predictor on top of these simpler classes.

AdaBoost offers a key advantage: it allows fine control over the tradeoff between approximation error and estimation error by adjusting a single parameter. This parameter manipulation is a powerful tool to adjust the model's behavior according to the specific requirements of the problem.

Boosting involves enhancing the capabilities of linear predictors by combining them with other functions. AdaBoost has been successfully employed in tasks like facial detection in images, showcasing how a theoretical concept can lead to practical solutions with substantial real-world significance.

###### Approximation Error
1. Approximation error is related to how well a chosen model (hypothesis) can approximate the true underlying relationship between the input data and the target output.
2. It's the error that occurs when a model with limited complexity or expressiveness is unable to accurately capture all the complexities and nuances present in the data distribution.
3. This error arises because real-world data distributions can be quite complex, and a simple model may not be able to represent them accurately.
4. The approximation error is usually higher when using simple models that cannot represent intricate patterns in the data.

###### Estimation Error
1. Estimation error is a result of the inherent uncertainty in estimating model parameters from finite training data.
2. It represents how much the learned model's parameters deviate from the true parameters of the underlying data distribution.
3. Estimation error arises because the model's parameters are learned from a limited sample of data, which may not fully capture the true characteristics of the entire data distribution.
4. More complex models tend to have higher estimation error because they have more parameters that need to be estimated from the available data.

In essence, approximation error is about the model's ability to fit the data distribution accurately, while estimation error is about the uncertainty in estimating the model's parameters from limited data. Both errors contribute to the overall error of a machine learning model. The goal of machine learning algorithms is to find a balance between these two types of errors to achieve the best possible predictive performance on new, unseen data.

Estimation error is a bit like when we're trying to figure out something, but there's some noisy or confusing information in our data. It's like if we're playing a game and someone is making funny sounds in the background – that noise can make it harder for us to focus and make a good guess.

In the same way, when we're teaching a computer to learn from data, sometimes there might be some weird or wrong data that makes it hard for the computer to learn perfectly. That's the estimation error – it's like the computer's guess is a bit off because of that noise or outliers in the data.

Approximation error is a bit like when we're learning something new, and we start with the basics. If we're learning to add single-digit numbers, that's easy. But when we move to adding two-digit numbers, especially when we have to carry numbers over, it gets a bit trickier.

So, in the beginning, our answer might not be perfect because we're still learning and we're using a simpler method. But as we practice and get better at carrying numbers over, our answers become more accurate. That initial not-perfect answer is like the approximation error – it's close, but not exactly right because we're using a simpler way to solve a more complex problem. (single digit addition model - simple model; carry concept - we are climbing one level up).

###### History
AdaBoost's origins trace back to a theoretical question posed by Kearns and Valiant in 1988: Can a weak learner, which is an algorithm that performs slightly better than random guessing, be "boosted" to become a strong learner, which is a highly accurate algorithm? This question remained unsolved until 1990 when Robert Schapire, a graduate student at MIT, provided a solution. However, the initial proposed solution wasn't very practical in real-world applications.

In 1995, Schapire, along with Yoav Freund, introduced the AdaBoost algorithm. AdaBoost was a groundbreaking development because it was the first practical implementation of boosting. This algorithm was simple yet elegant in its approach and quickly gained enormous popularity within the machine learning community. The work of Freund and Schapire received significant recognition and numerous awards for their contributions to the field.

## PAC Learning in brief
Probably Approximately Correct Learning - A theoretical framework in machine learning that helps us understand how well a learning algorithm can generalize from a limited set of examples. It's like trying to figure out how well a computer program can learn from a bunch of examples to make predictions on new, unseen data.

In PAC learning, we have a few key components:
1. Hypothesis Class (H): This is the set of all possible hypotheses or predictions that our learning algorithm can make. Think of it as all the different ways the computer can learn and make predictions.
2. Sample Complexity (mH): This is a function that tells us how many examples the learning algorithm needs to see in order to learn well. It depends on factors like how complex the hypothesis class is and how accurate we want the learning to be.
3. Accuracy and Confidence (ε and δ): These are parameters that define how well we want the learning algorithm to perform. ε represents the maximum allowable error in the algorithm's predictions, and δ represents the probability that the algorithm's predictions won't deviate too far from reality.
4. Distribution (D) and Labeling (f): D is a description of how the examples are generated, like a set of rules for creating them. f is the true labeling function that assigns correct labels to the examples.

The PAC learning goal is this: Given the hypothesis class, the accuracy and confidence requirements, and the distribution of examples, we want to find a learning algorithm that can generate a hypothesis (prediction rule) that works well most of the time on new, unseen examples.

Here's how it works:
1. We gather a bunch of examples generated by D and labeled by f.
2. We use our learning algorithm with these examples to create a hypothesis h.
3. We measure how well h performs on new examples that weren't in the training set.

For the learning algorithm to be considered successful in the PAC framework, it must meet two conditions:
1. The error of the hypothesis on new examples should be close to ε (the allowable error) with high confidence (probability 1-δ).
2. The number of examples it needs (m) should be reasonable, not too large.

PAC learning helps us understand the fundamental trade-off between the complexity of the hypothesis class, the number of examples needed, and the desired accuracy of predictions. It's a powerful theoretical concept that gives us insights into the capabilities and limitations of learning algorithms.

## PAC Learning with an example
Imagine we're teaching a robot to distinguish between cats and dogs based on pictures. We have a magical book of animal pictures with labels that say "Cat" or "Dog." Our goal is to teach the robot to look at new pictures and tell us whether it's a cat or a dog.

1. Hypothesis Class (H): The robot's "brain" is like its hypothesis class. It can think in different ways to make predictions—maybe it looks for pointy ears, long tails, or furry bodies.
2. Sample Complexity (mH): How many pictures should we show the robot so it learns well? If it's a simple robot, maybe it needs just a few examples. If it's a complex robot, it might need more pictures to learn accurately.
3. Accuracy and Confidence (ε and δ): How accurate do we want the robot to be? Maybe we're okay if it's wrong sometimes, as long as it's not too often (ε). And we want to be pretty confident in its answers, so we set a high probability (1-δ).
4. Distribution (D) and Labeling (f): D is how the pictures are taken, like if they're mostly daytime pictures or nighttime pictures. f is the true labels, like "Cat" or "Dog" for each picture.

Imagine we have a robot that can only make very basic decisions. It looks at a picture and can only say "Cat" or "Dog" based on whether it sees a lot of fur or not much fur. This robot is simple because it uses only one straightforward rule to decide. For a simple robot, we might not need to show it too many pictures before it figures out the basic rule. So, its sample complexity (mH) is relatively low.

Imagine a robot can analyze many different features of the pictures: the shape of the ears, the color of the eyes, the length of the tail, and more. It uses all these different features to decide whether it's looking at a cat or a dog. For a complex robot, it might take more examples (pictures) for it to understand all these different features and make accurate predictions. So, its sample complexity (mH) is higher compared to the simple robot.

###### PAC Learning Process:
1. We pick a bunch of pictures from our magical book (generated by D) and put labels on them (done by f).
2. We teach the robot using these pictures. It looks at the pictures and learns what makes a cat different from a dog.
3. Now we show the robot new pictures it hasn't seen before. It uses what it learned to guess if each picture is a cat or a dog.

###### PAC Learning Success:
1. The robot's guesses should be pretty close to reality (ε) most of the time with high confidence (1-δ). This means it's learning well and can tell cats from dogs even in new pictures.
2. The number of pictures you showed the robot (m) should be reasonable. If you had to show it a million pictures for it to work, that's not very practical!

PAC learnable - if we can teach the computer to be really good at it.

## Fundamental Theorem of Learning Theory
Any concept class (a set of possible hypotheses) that is PAC learnable can be learned using the ERM (Empirical Risk Minimization) algorithm. This means that if a class of concepts can be learned from a sample of data in a way that ensures good generalization to new data, then the ERM algorithm can be used to find a hypothesis that accurately represents that concept class.

###### ERM Algorithm (Empirical Risk Minimization)
A fundamental approach in machine learning. It's used to find the hypothesis (model) that minimizes the empirical risk, which is the average error of the hypothesis on a training dataset. In other words, ERM tries to find the hypothesis that makes the fewest mistakes on the training data. The hope is that this hypothesis will also perform well on unseen data.

###### Computational hardness 
A situations where a computational task is extremely difficult or time-consuming for a computer to perform. In the context of ERM, the computational hardness arises when the process of minimizing the empirical risk becomes very complex due to the complexity of the hypothesis space (the set of possible hypotheses) or the size of the data.

Imagine we have a very large dataset and our hypothesis space is very complex, like trying to fit a highly intricate pattern to a massive amount of data points. This task can become computationally infeasible, meaning it would take an impractical amount of time and resources for a computer to find the best hypothesis.

###### Why ERM Can Be Hard:
1. Complex Hypothesis Space: If the hypothesis space is very large and complex, searching through all possible hypotheses to find the best one can be extremely time-consuming. Think of trying to search for the best-fitting model in a space with billions of possibilities.
2. High-Dimensional Data: When the data has a high number of dimensions (features), the complexity of the search increases. This is known as the "curse of dimensionality."
3. Non-Convex Optimization: The ERM optimization process can involve finding the minimum of a function, and if that function is non-convex (has multiple local minima), it can be difficult to ensure that the algorithm finds the global minimum.
4. Computational Resources: The more complex the task, the more computational power and time it might require. In cases where the computation demands exceed available resources, the ERM process becomes infeasible.

## Weak Learnability
Imagine we have a magic learning machine, and we want to teach it to recognize cats from dogs based on pictures. But this magic machine isn't perfect; sometimes it might make mistakes. Now, there's a way to measure how good this magic machine is at learning. We'll call it "$\gamma$-Weak-Learnability."

1. Magic Learning Machine's Performance: We want the magic machine to be better than random guessing, but not necessarily perfect. Let's say a random guess gets it right about half the time (which makes sense because there are two choices - cat or dog). But we want the magic machine to do a little better than that.
2. $\gamma$ (Gamma) Value: We can set a goal for how much better than random guessing we want the magic machine to be. This goal is given by a number called "$\gamma$" (gamma). If $\gamma$ is small, it means we want the magic machine to be just a tiny bit better than random guessing. If $\gamma$ is larger, we want it to be significantly better.
3. Weak Learning Condition: Now, if this magic machine can learn in a way that, most of the time, its mistakes are fewer than what random guessing would make, then we say it's a "$\gamma$-Weak-Learner."
4. Learning Class: And if this magic machine can do this not just for cats and dogs, but for many different things we want it to learn about, then we say that the things it can learn about belong to a "$\gamma$-Weak-Learnable" class.

In simpler terms, $\gamma$-Weak-Learnability says that the magic machine doesn't have to be perfect, but it needs to be better than random guessing. And if it can consistently do this for various things you want it to learn, then you can call those things "$\gamma$-Weak-Learnable."

So, $\gamma$-Weak-Learnability is a way to measure how well a learning machine can do when it's not required to be exactly right all the time, but just a bit better than random guessing.

###### Definition
A learning algorithm, denoted as A, is considered a $\gamma$-weak-learner for a hypothesis class H if there exists a function mH : (0, 1) → N (a function that takes values from the interval (0, 1) and maps them to positive integers) such that the following conditions are met:
1. For any $\delta$ (a value between 0 and 1), any distribution D over a set of data instances X, and any labeling function f that assigns labels of {±1} to instances in X, if the realizable assumption holds (meaning there's a hypothesis in H that can perfectly fit the data), then when algorithm A is run on m ≥ mH($\delta$) independent and identically distributed (i.i.d.) examples generated by distribution D and labeled by function f, the algorithm returns a hypothesis h.

$\delta$ (Delta): a small value between 0 and 1, representing the level of confidence we want in the statement. For example, if δ is 0.1, we want to be 90% confident in the statement.</br>
Distribution D: how the data instances are spread out or distributed. For example, if we're teaching a machine to recognize animals, D could represent how likely different animals are to appear in our training data.</br>
Labeling function f: assigns labels of either +1 or -1 to the instances in our data. In the case of animals, it could be +1 for dogs and -1 for cats.</br>
Realizable Assumption: there's a hypothesis (a guess or prediction) in the hypothesis class H that can perfectly match the data. In our case, it's like saying there's a way to teach the machine so it can perfectly tell the difference between cats and dogs.</br>
Algorithm A: like our magic machine, that's trying to learn from the data.</br>
m ≥ mH(δ): we're using at least a certain number of data examples. The value mH($\delta$) depends on the complexity of the hypothesis class H and the confidence level $\delta$.</br>
Independent and Identically Distributed (i.i.d.): the examples are randomly and fairly picked, and they're not related to each other. For instance, each picture of a cat or dog is randomly chosen, and they're not influenced by each other.</br>

So, in simpler terms, this statement is saying that if we have a way to perfectly match our data using the hypothesis class H, and we use an algorithm A to learn from enough examples generated by distribution D and labeled by f, then the algorithm A will return a guess (hypothesis) about the data. The idea is that with enough good-quality examples, the algorithm can learn well.

2. This hypothesis h should have the property that, with a probability of at least 1 − $\delta$, its error rate (L(D,f)(h), which is the fraction of instances on which h disagrees with f) is less than or equal to 1/2 − $\gamma$.</br>

Hypothesis h: Tguess or prediction made by the learning algorithm A based on the examples it was given.</br>
Probability of at least 1 - δ: we want this statement to be true with a very high probability. If $\delta$ is small (close to 0), then 1 - $\delta$ is very close to 1, meaning we're really confident.</br>
Error rate L(D,f)(h): how often the guess h made by the algorithm disagrees with the labeling function f on the instances in our data. If h and f agree on most instances, the error rate is low; if they disagree a lot, the error rate is high.</br>
1/2 - $\gamma$: This is a value less than 1/2 (which is like guessing randomly). It's like saying "a little better than random guessing." $\gamma$ (gamma) measures how much better than random guessing the algorithm's guess needs to be.

So, this statement is saying that the guess (hypothesis) the algorithm makes should be right most of the time. It's okay if it's wrong sometimes, but with a very high probability (1 - $\delta$), the error rate of the guess should be less than or equal to a certain value (1/2 - $\gamma$). In other words, the algorithm's guess should be better than random guessing, and we're really confident that this is the case.

In essence, $\gamma$-Weak-Learnability is a relaxed version of PAC (probably approximately correct) learning. It addresses the situation where we don't require the algorithm to find an arbitrarily accurate classifier (as in strong learnability). Instead, we aim for a hypothesis that performs slightly better than random guessing, with an error rate of at most 1/2 − $\gamma$.

The relationship between VC dimension, sample complexity, and weak learnability suggests that weak learning is as challenging as strong learning from a statistical perspective. However, considering computational complexity, the use of a simple hypothesis class and ERM could make weak learning more feasible and practical.








