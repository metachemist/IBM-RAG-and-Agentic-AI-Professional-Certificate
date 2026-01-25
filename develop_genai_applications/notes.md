## Retrieval Augmented Generation, or RAG.
    Enhances AI's ability to provide accurate, context-aware responses by integrating real-time information retrieval.

## Multimodal AI 
    is another advancement that allows systems to process and integrate various types of data – text, images, audio, and video – enabling more dynamic and interactive user experiences.

## Agentic AI
    Represents a further shift, equipping systems with the ability to reason, plan, and autonomously execute tasks. 

    ![Course Outline](image.png)


----------------------------------------------------


## 1. Definition of Generative AI
The video defines Generative AI (GenAI) as a subset of artificial intelligence capable of creating new content—such as text, images, audio, video, and code—rather than simply analyzing or classifying existing data. It contrasts this with Discriminative AI (traditional AI), which is typically used to classify data (e.g., distinguishing between a cat and a dog) or predict numeric values.

### 2. How It Works
Foundation Models: The lecture explains that GenAI relies on "foundation models"—large-scale models pre-trained on vast amounts of unstructured data.

Pattern Learning: Instead of being programmed with explicit rules, these models learn patterns and structures from their training data. When given a prompt, they use these learned patterns to generate new, unique outputs that resemble the training data but are not identical copies.

Next-Token Prediction: For text models (Large Language Models or LLMs), the video often explains the mechanism of predicting the "next most likely word" (token) to generate coherent sentences.

## 3. Key Concepts & Terminology
Prompts: The input provided by the user (text, instruction, or code) that guides the model's output. The quality of the output depends heavily on the quality of the prompt.

Hallucinations: The video typically warns about "hallucinations," where the model generates confident but factually incorrect or nonsensical information because it is prioritizing pattern completion over factual accuracy.

## 4. Applications and Capabilities
The lecture outlines various real-world applications to illustrate the technology's versatility as a "general-purpose technology":

Text Generation: Writing emails, essays, summaries, and creative stories.

Code Generation: Assisting developers by writing or debugging code snippets.

Image/Media Creation: Generating visual art or realistic images from text descriptions.


--------------------------------------------------

**Lecture:** What are Generative AI Models?

## 1. The Core Distinction: Discriminative vs. Generative
To understand GenAI, you must distinguish it from the "Classical ML" models you have likely studied previously.

### Discriminative Models (The "Validators")
* **Function:** Classifies input data into pre-defined categories. It draws a decision boundary between classes.
* **The Math:** Learns the **Conditional Probability Distribution**: $P(Y|X)$
    * *Query:* Given input $X$ (e.g., an image), what is the probability it belongs to class $Y$ (e.g., "Cat")?
* **Analogy:** A teacher grading a test (Pass/Fail) or a fraud detection system (Fraud/Not Fraud).
* **CS Context:** SVMs, Logistic Regression, Random Forests, Standard CNNs.

### Generative Models (The "Creators")
* **Function:** Creates new data instances that resemble the training data. It captures the underlying structure of the data.
* **The Math:** Learns the **Joint Probability Distribution**: $P(X,Y)$ (or $P(X)$ in unsupervised settings)
    * *Query:* How likely is a given sample $X$ to occur in the dataset? (Allows sampling new $X$ values).
* **Analogy:** An artist studying thousands of portraits to paint a totally new person who looks like they belong in that era.
* **CS Context:** GANs, VAEs, Transformers (LLMs).

----------------------------------------------------

