# Overview

In Session 4, we dig into LangChain and LangGraph!  LangChain has emerged as the leader of the pack of infrastructure [orchestration tools](https://github.com/a16z-infra/llm-app-stack?tab=readme-ov-file#orchestrators).

> LangChain is a framework for developing applications powered by large language models (LLMs). [[Ref](https://python.langchain.com/docs/introduction/)]
> 

LangChain, as a company, has built an entire ecosystem of tools (we’ll often refer to this as the Lang-X ecosystem).  

During this session, we’ll learn about LangChain, LangGraph, and LangSmith. LangChain and LangGraph are used for orchestration (we can think of LangGraph as the next evolution of LangChain, well-suited for agents and agentic reasoning loops). 

> LangGraph is built for developers who want to build powerful, adaptable AI agents. [[Ref](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)]
> 

LangSmith, on the other hand, is used for monitoring, visibility, logging, and evaluation. 

> **LangSmith** is a platform for building production-grade LLM applications. [[Ref](https://docs.smith.langchain.com/)]
> 

During Session 4, we will also cover a specific use case that we’ll build on in the next session. From prototype to production with LangChain, let’s gooooooo!

![image.png](attachment:f192457e-bff6-4ede-a7f6-4a088a77065c:image.png)

# 📛 Required **Tooling & Account Setup**

1. Create an ollama account
    
    [Sign in to Ollama](https://ollama.com/signin)
    
2. [Optional] Create a LangSmith account 👇
    
    [LangSmith](https://smith.langchain.com/)
    

# 🧑‍💻 Recommended Pre-Work!

In this class, we’ll learn each of the key constructs of LangChain, and we’ll build and evaluate a RAG application with LangChain. Here’s what to do to prepare:

1. 🏫 During this session, we’ll begin exploration of our cohort’s use case, inspired by the many Deep Research applications that every model provider has rolled out in 2025!
2. The purpose of LangGraph (and subsequently, [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/)) becomes apparent the more you study the idea of state machines. As an introduction, check out CEO Harrison Chase’s TED Talk on Cognitive architectures.
    
    [The magical AI assistants of the future — and the engineering behind them](https://www.ted.com/talks/harrison_chase_the_magical_ai_assistants_of_the_future_and_the_engineering_behind_them?subtitle=en)
    
3. 🤓 Check out additional reading material in [Go Deeper](https://www.notion.so/Session-4-RAG-with-LangGraph-OSS-Local-Models-Eval-w-LangSmith-26acd547af3d80838d5beba464d7e701?pvs=21).

# ⛓️ Core Constructs: LangChain

When orchestrating complex LLM applications that leverage context and reasoning,  LangChain has emerged as a best-practice tool.  To begin, it’s instructive to start with how we can leverage the pattern of Retrieval Augmented Generation to build apps using LangChain.

What are the core components we need to understand?

Let us start, where we should, with the abstraction of the `chain` in LangChain.

<aside>
⛓️ Chain: a sequence of calls to other components

</aside>

With this idea of chains in mind, we can understand that `LangChain Expression Language, or LCEL`, allows us to ***compose chains*** easily.  LCEL also enables us to build production-grade prototypes that can be deployed with no code changes.

**A Simple Chain - Chat Models and Prompts**

The first chain we’ll build with LCEL combines `Model` and `Prompt` from Models I/O.

Using LLM models that have been instruction-tuned and fine-tuned for chat is a best practice for LLM-powered applications.  Just as we learned about the {System, User, Assistant} roles in OpenAI, we can leverage chat-style models using the {System, Human, AI} roles in LangChain.  These are, for all intents and purposes, the same.

![Chat completions roles, OpenAI vs. LangChain.](https://prod-files-secure.s3.us-west-2.amazonaws.com/d7a211dd-f900-47ac-8e0e-f6b027cb71b3/ce19a888-4a46-4d1a-8ab2-814077530b32/image.png)

Chat completions roles, OpenAI vs. LangChain.

*** Recall that with o1 models and newer, `developer` messages replace previous `system`messages.*

Building our first LLM chain requires a chat prompt template.  We have seen this in our Pythonic RAG system, and can see many other examples [here](https://python.langchain.com/docs/how_to/#prompt-templates).  Here is what our first chain looks like in code.

> `chain = chat_prompt | openai_chat_model`
> 

**Our Second Chain - Retrieval Augmented Generation**

To do RAG with LangChain, we leverage the same process we used to build our Pythonic RAG application.  Below these steps are outlined with the relevant LangChain constructs and documentation that we’ll use during Session 4!

1. Create **Database**
    1. `Document Loader` [[Ref](https://python.langchain.com/docs/how_to/#document-loaders)]
    2. `Text Splitter` [[Ref](https://python.langchain.com/docs/how_to/#text-splitters)]
    3. `Embedding Model` [[Ref](https://python.langchain.com/docs/integrations/text_embedding/)]
    4. `Vector Store` [[Ref](https://python.langchain.com/docs/how_to/#vector-stores)]
2. Ask **Question**
    1. `Embedding Model` [[Ref](https://python.langchain.com/docs/integrations/text_embedding/)]
3. Find **References**
    1. `Retriever` [[Ref](https://python.langchain.com/docs/how_to/#retrievers)]
4. **Augment** the Prompt
    1. `Prompt` [[Ref](https://python.langchain.com/docs/how_to/#prompt-templates)]
5. **Generate** a better answer!
    1. `Model` [[Ref](https://python.langchain.com/docs/integrations/chat/)]

<aside>
⛓️ In the end, we want to create a single RAG chain that leverages LCEL

</aside>

Here’s an example in code of such a RAG chain👇

```python
simple_rag  = (
    {"context": retriever, "query": RunnablePassthrough()}
    | chat_prompt
    | openai_chat_model
    | StrOutputParser()
)
```

Notice that we see the word “Runnable” in the code.

# **The Runnable**

 In LangChain, a Runnable is like a LEGO brick in your AI application - it's a standardized component that can be easily connected with other components. The real power of Runnables comes from their ability to be combined in flexible ways using LCEL (LangChain Expression Language).

*Every component of a chain is a **runnable**.*  

In fact, **the primary abstraction in the LangChain ecosystem is the runnable**.

> We often refer to a `Runnable` created using LCEL as a "chain" [[Ref](https://python.langchain.com/docs/concepts/lcel/)]
> 

## **Key Features of Runnables**

**1. Universal Interface**

Every Runnable in LangChain follows the same pattern:

- Takes an input
- Performs some operation
- Returns an output

This consistency means you can treat different components (like models, retrievers, or parsers) in the same way.

**2. Built-in Parallelization**

Runnables come with methods for handling multiple inputs efficiently:

```python
# Process inputs in parallel, maintain order
results = chain.batch([input1, input2, input3])

# Process inputs as they complete
for result in chain.batch_as_completed([input1, input2, input3]):
    print(result)
```

**3. Streaming Support**

Perfect for responsive applications:

```python
# Stream outputs as they're generated
for chunk in chain.stream({"query": "Tell me a story"}):
    print(chunk, end="", flush=True)
```

**4. Easy Composition**

The `|` operator makes building pipelines intuitive, as seen above! e.g.;

```python
# Create a basic RAG chain
rag_chain = retriever | prompt | model | output_parser
```

**Common Types of Runnables**

- Language Models: Like our `ChatOpenAI` instance
- Prompt Templates: Format inputs consistently
- Retrievers: Get relevant context from a vector store
- Output Parsers: Structure model outputs
- LangGraph Nodes: Individual components in our graph

Think of Runnables as the building blocks of your LLM application. Just like how you can combine LEGO bricks in countless ways, you can mix and match Runnables to create increasingly sophisticated applications!

# 🕸️  Core Constructs: LangGraph

Instead of building our entire RAG chain using runnables, the best-practice in 2025 is to use LangGraph directly.

<aside>
🔄 LangGraph lets us **add cycles** to applications built on LangChain.

</aside>

The essence of LangGraph is that it uses graphs to add cycles.  

**Why Cycles?**

We can think of a cycle in our graph as a more robust and customizable loop. It allows us to keep our application ***agent-forward*** while still giving the powerful functionality of traditional loops.

Due to the inclusion of cycles over loops, we can also compose rather complex flows through our graph in a much more readable and natural fashion. Effectively allowing us to recreate application flowcharts in code in an almost 1-to-1 fashion.

**Why LangGraph?**

During this session, *we will be using LangGraph as a Directed Acyclic Graph (DAG*).  Beyond the agent-forward approach - we can easily compose and combine traditional DAG chains with powerful cyclic behavior due to the tight integration with LCEL. 

In this way, LangGraph is a natural extension to LangChain's core offerings!

## Graphs

Graphs are collections of connected objects: nodes and edges.

- **Node**: Think `function` or `runnable`; i.e. *something that changes **state***
- **Edge**: Think path to take; i.e., *where to pass **state** object next*

A state object is initially defined by passing a state definition to a class representing the graph.  This state object, or `StateGraph`, gets updated over time.  The agent's internal state is represented simply as a list of messages.  Remember how we interacted with the OpenAI API with a list of messages with roles?  Same idea.

Just as every component of a chain is a runnable, each node in our graph can be a runnable, or even an entire chain!  

Welcome to the next layer of abstraction.

# ⚒️ LangSmith for Evaluation & Visibility

> **LangSmith** is a platform for building production-grade LLM applications. [[Ref](https://docs.smith.langchain.com/)]
> 

In addition to building out a RAG application with LangChain, we will also learn what the developer sees behind the scenes when users interact with our LangChain application!  Enter … LangSmith.

> “LangSmith turns LLM “magic” into enterprise-ready applications.”
> 

Ultimately, we will use LangSmith for evaluation and for visibility.

**Evaluation**

The key to understanding the right way to do quantitative evaluation with tools like LangSmith is Metrics Driven Development (MDD). There are three steps to keep in mind:

1. Establish a baseline
2. **Change stuff** that potentially improves baseline
3. **Recalculate** metrics

It’s not about absolute values.  Rather, MDD is about *relative changes* in evaluation metrics.

There are three evaluator types [[Ref](https://docs.smith.langchain.com/old/evaluation/faq/evaluator-implementations)] we’ll look at in LangSmith, including:

1. QA evaluators
2. Criteria evaluators (no labels)
3. Criteria evaluators (with labels)

<aside>
💬 Each of the evaluations that we’ll leverage in LangSmith have been created *through direct prompting*

</aside>

- Ex. Chain of Thought QA evaluation with `cot_qa` , the source code 👇
    
    ```python
    You are a teacher grading a quiz.
    You are given a question, the context the question is about, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.
    Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.
    
    Example Format:
    QUESTION: question here
    CONTEXT: context the question is about here
    STUDENT ANSWER: student's answer here
    EXPLANATION: step by step reasoning here
    GRADE: CORRECT or INCORRECT here
    
    Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 
    ```
    

Can you spot the prompting best-practices at play in `cot_qa` prompt?

There are many criteria that we can decide to assess via LangSmith, from correctness, harmfulness, and relevance to controversially, misogyny and criminality.  Look at [some of their prompts](https://docs.smith.langchain.com/old/evaluation/faq/evaluator-implementations#criteria-evaluators-no-labels) and imagine what aspects would be the best for your next evaluation task! 

**Visibility**

When it comes to observing and assessing the performance of our application, in addition to evaluation metrics we should also be watching key production metrics including `token count` and `latency`. 

More generally, visibility is all about seeing inside our application and being able to drill down into problems as soon as they become obvious.

[Tracing](https://docs.smith.langchain.com/concepts/tracing) allows us to keep track of everything throughout each piece of our LLM application pipeline (e.g., chain!).  A run represents a single unit of work or operation within your LLM application, and a trace is a collection of runs! 

# 🕳️ Go Deeper!

- 🤖 For today’s primary build, we’ll be leveraging the open-source [gpt-oss](https://openai.com/index/introducing-gpt-oss/) model (August, 2025) that we did a deep dive on [here](https://www.youtube.com/live/nyb3TnUkwE8?si=Yz97mSOltslcPnIM).
- https://blog.langchain.com/the-rise-of-context-engineering/ (June 2025)
- [Is LangGraph used in Production?](https://blog.langchain.com/is-langgraph-used-in-production/) (Feb 2025)