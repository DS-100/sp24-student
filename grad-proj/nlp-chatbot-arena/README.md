# Data200 Gradaute Project: Data Science with Natural Language Processing

A commont task in real life data analysis involves working with text data.
In this project, we will work with a dataset consisting natural language questions asked by human and answers provided by chatbots.

The goal of this project is to:

- Prepare you to work with text data by learning common techniques like embedding generation, tokenization, and topic modeling.
- Work with real world data in its targetted domain. The data is non-trivial in both size and complexity.
- Ask open ended questions and answer them using data at hand.

This documents contains the following sections:
* [Dataset Description](#dataset-description)
* [Project Tasks](#project-tasks)
* [Getting Started](#getting-started)
* [Resources](#resources)

## Dataset Description

The source dataset comes from https://huggingface.co/datasets/lmsys/chatbot_arena_conversations. The author describes the dataset as follows:

> This dataset contains 33K cleaned conversations with pairwise human preferences. It is collected from 13K unique IP addresses on the Chatbot Arena from April to June 2023. Each sample includes a question ID, two model names, their full conversation text in OpenAI API JSON format, the user vote, the anonymized user ID, the detected language tag, the OpenAI moderation API tag, the additional toxic tag, and the timestamp.

[Chatbot Arena](https://chat.lmsys.org/) is a platform where users can ask questions and two chatbots will provide answers. The user then votes on which chatbot provided the best answer. The dataset contains the questions, the answers, and the user votes.

You can find the processed dataset in `./chatbot-arena-conversations.jsonl.gz`. The dataset is in JSON line format and compressed using gzip. It has gone through the following preprocessing steps to make analysis easier:

- Removed non-English conversations.
- Removed conversations with more than one round.
- Removed conversations classified as toxic or harmful.

The dataset you will be working with contains `25322` rows (out of `33000` total rows) and `7` columns. The columns are:

- `question_id`: A unique identifier for the question.
- `model_a`: The name of the first chatbot model.
- `model_b`: The name of the second chatbot model.
- `winner`: The name of the chatbot model that won the user vote.
- `judge`: The anonymized user ID that voted.
- `conversation_a`: The conversation between the user and `model_a`.
- `conversation_b`: The conversation between the user and `model_b`.

An example of the data is as follows:

```json
{
  "question_id": "c0fc42c6f5f14f2aa5a89f71f8553730",
  "model_a": "vicuna-13b",
  "model_b": "koala-13b",
  "winner": "model_a",
  "judge": "arena_user_973",
  "conversation_a": [
    {
      "content": "Count from 1 to 10 with step = 3",
      "role": "user"
    },
    {
      "content": "1, 4, 7, 10\n\nCounting with a step of 3 means that you will increase the number by 3 each time. So, the first number is 1, the next is 4 (1 + 3), the next is 7 (4 + 3), and so on.",
      "role": "assistant"
    }
  ],
  "conversation_b": [
    {
      "content": "Count from 1 to 10 with step = 3",
      "role": "user"
    },
    {
      "content": "1, 4, 7, 10",
      "role": "assistant"
    }
  ]
}
```

There are two axuliary dataset that you can use to help with your analysis:

- `./chatbot-arena-prompts-embeddings.npy` contains the 256 dimensional text embeddings for each of the human question. The embeddings are generated using OpenAI's `text-embedding` model. We will explain what is embedding and how can you use them later in this notebook. It has shape (25322, 256) and have dtype('float32').
- `./chatbot-arena-gpt3-scores.jsonl.gz` contains labels for the dataset you can use for later modeling tasks. It has the following fields:
  - `question_id`: The unique identifier for the question, as seen in `./chatbot-arena-conversations.jsonl.gz`.
  - `prompt`: The extracted human question. This is equivalent to the first message in `conversation_a` and `conversation_b` in `./chatbot-arena-conversations.jsonl.gz`.
  - `openai_scores_raw_choices_nested`: The response from OpenAI GPT 3.5 model (see later for the prompt). It contains the evaluated topic model, reason for a hardness score from 1 to 10, and the value. For each prompt, we have 3 responses. We extracted the fields into the following columns.
  - `topic_modeling_1`, `topic_modeling_2`, `topic_modeling_3`: The topic modeling for the first, second, and third response. Each topic should have two words.
  - `score_reason_1`, `score_reason_2`, `score_reason_3`: The reason for the hardness score for the first, second, and third response.
  - `score_value_1`, `score_value_2`, `score_value_3`: The hardness score for the first, second, and third response.

```json
{
  "question_id": "58210e39b3fd4441a2bd4a518bb44c2d",
  "prompt": "What is the difference between OpenCL and CUDA?",
  "openai_scores_raw_choices_nested": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "{\n    \"topic_modeling\": \"Technical Comparison\",\n    \"score_reason\": \"This prompt requires the AI to accurately compare and contrast two distinct technologies, OpenCL and CUDA. It assesses the AI's factual accuracy and knowledge of these technologies, as well as its ability to articulate the differences between them.\",\n    \"score_value\": 9\n}",
        "role": "assistant",
        "function_call": null,
        "tool_calls": null
      }
    },
    {
      "finish_reason": "stop",
      "index": 1,
      "logprobs": null,
      "message": {
        "content": "{\n    \"topic_modeling\": \"Software Comparison\",\n    \"score_reason\": \"This prompt assesses the AI's factual accuracy in distinguishing between two similar but distinct software frameworks.\",\n    \"score_value\": 8\n}",
        "role": "assistant",
        "function_call": null,
        "tool_calls": null
      }
    },
    {
      "finish_reason": "stop",
      "index": 2,
      "logprobs": null,
      "message": {
        "content": "{\n    \"topic_modeling\": \"Comparison, Technology\",\n    \"score_reason\": \"This prompt requires the AI to demonstrate knowledge of two different technologies, compare their features, and explain their distinctions. This task assesses the AI's factual accuracy and proficiency in understanding complex technological concepts.\",\n    \"score_value\": 9\n}",
        "role": "assistant",
        "function_call": null,
        "tool_calls": null
      }
    }
  ],
  "topic_modeling_1": "Technical Comparison",
  "score_reason_1": "This prompt requires the AI to accurately compare and contrast two distinct technologies, OpenCL and CUDA. It assesses the AI's factual accuracy and knowledge of these technologies, as well as its ability to articulate the differences between them.",
  "score_value_1": 9,
  "topic_modeling_2": "Software Comparison",
  "score_reason_2": "This prompt assesses the AI's factual accuracy in distinguishing between two similar but distinct software frameworks.",
  "score_value_2": 8,
  "topic_modeling_3": "Comparison, Technology",
  "score_reason_3": "This prompt requires the AI to demonstrate knowledge of two different technologies, compare their features, and explain their distinctions. This task assesses the AI's factual accuracy and proficiency in understanding complex technological concepts.",
  "score_value_3": 9
}
```

#### The prompt used to generate ground truth labels

```markdown
We are interested in understanding how well the following input prompts can evaluate an
AI assistant’s proficiency in problem-solving ability, creativity, or adherence to real-world
facts. Your task is to assess each prompt based on its potential to gauge the AI’s capabilities
effectively in these areas.

For each prompt, carry out the following steps:

1. Topic Modeling: Use two words to describe the task intended.
2. Assess the Potential: Consider how challenging the prompt is, and how well it can
   assess an AI’s problem-solving skills, creativity, or factual accuracy. Briefly explain your
   reasoning.
3. Assign a Score: Assign a score on a scale of 1 to 10, with a higher score representing
   a higher potential to evaluate the AI assistant’s proficiency effectively. Use double square
   brackets to format your scores, like so: [[5]].

Guidelines for Scoring:
• High Score (8-10): Reserved for prompts that are particularly challenging and excellently designed to assess AI proficiency.
• Medium Score (4-7): Given to prompts that have a moderate potential to assess the AI’s
capabilities.
• Low Score (1-3): Allocated to prompts that are either too easy, ambiguous, or do not
adequately assess the AI’s capabilities.

Ensure to critically evaluate each prompt and avoid giving high scores to prompts that are
ambiguous or too straightforward.

The output MUST follow a JSON format:
{
"topic_modeling": "...",
"score_reason": "...",
"score_value": ...
}
```

#### Generate your own embeddings
Note that the provided embeddings are generated using OpenAI's model, which means if you want generate new prompt, you needs an OpenAI developer account ($5 free credit for new signup otherwise needing a credit card). The code used to generate the embeddings is as follows:

```python
import openai
client = openai.OpenAI()
client.embeddings.create(
    model="text-embedding-3-small", input=prompt, extra_body=dict(dimensions=256)
)
```

You are welcomed to use open source models (you can use huggingface on datahub!) to generate the embeddings for the new prompts. A good start would be using huggingface pipeline for embedding generation:

```python
from transformers import pipeline

embedder = pipeline("feature-extraction", model="distilbert-base-uncased")
embeddings = embedder(prompt)
```

You can see more in the [feature extraction pipeline documentation](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/pipelines#transformers.FeatureExtractionPipeline). It will result in dimension of 768 for each prompt.

## Project Tasks

Your tasks will be open ended and feel free to explore the data as you see fit. Overall, you should aim to answer the following questions:

- EDA Tasks: Tell us more about the data. What do you see in the data? Come up with questions and answer about them. For example, what are the win-rate of GPT4? What are the most common topics? Do different judges have different preferences?
- Modeling Tasks: Perform some modeling tasks given our ground truth labels. Can you train a logistic regression model to predict the winner given embeddings? How about a K-means clustering model to cluster the questions? Can you use linear regression to predict the hardness score?
- Analysis Tasks: By leveraging the question embeddings, can we find similar questions? How "repeated" is the questions in the dataset? Can you reproduce the Elo score rating for the chatbots?

For EDA task, we expect plots and story telling. For modeling tasks, we expect you to demostrate how the well model works and how to evaluate them. The analysis task is more open ended.


## Getting Started

To get started, we provide a notebook `nlp-chatbot-starter.ipynb` that demostrate how to load and inspect the data. The project is expected to be open ended!

## Resources

- [Joey's EDA and Elo rating modeling](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH) is a great resource to get started with the EDA. Note that (1) the plot is made with plotly, we recommend you to reproduce the plot with matplotlib or seaborn, and (2) the Elo rating is a good modeling task to reproduce but we expect you to do more than just that (for example, demostrate how Elo rating works and how to calculate it in your report).

- [An intuitive introduction to text embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/) is a good resource to understand what is text embeddings and how to use them.

- [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system) and [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) are essential to model a ranking among the pair wise comparison.

- [Huggingface pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) have many implementations of common NLP task for you to use, including sentiment analysis, summarization, text classification, etc.

- [spaCy](https://spacy.io/usage/spacy-101) is a wonderful library containing classifical NLP tasks like tokenization, lemmatization, etc.