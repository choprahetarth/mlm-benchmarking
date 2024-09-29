naive_instruction_with_image_start =  """Solve a question answering task, by looking at the given image, doing an ImageObserve, with interleaving Thought, Action, Observation steps. Use the provided image as context required to solve the question. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples."""

naive_instructions_default = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples."""


naive_instructions_with_image_tool = """Solve a question answering task with interleaving Thought, Action, Observation. Thought can reason about the current situation, and Action can be four types: 
(1) ImageObserve[image], which returns an in-depth explanation of the image.
(2) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(4) Finish[answer], which returns the answer and finishes the task.
Here are some examples."""

single_answer_image = """Solve a question answering task with interleaving Thought, Action, Observation. Thought can reason about the current situation, and Action can be four types: 
(1) ImageObserve[image], which returns an in-depth explanation of the image, in a single paragraph, without using bullet points or numbers.
(2) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(4) Finish[answer], which returns the answer and finishes the task.
Please return the answer as a single entity, how an MCQ is supposed to be solved, and not a phrase or multiple outputs.
Here are some examples."""

single_answer_image_cleaned = """Solve a question answering task with interleaving Thought, Action, Observation. Thought can reason about the current situation, and Action can be four types: 
(1) ImageObserve[image], which returns an in-depth explanation of the image, in a single paragraph.
(2) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(4) Finish[answer], which returns the answer and finishes the task.
Please return the answer as a single entity and not a phrase or multiple outputs.
Here are some examples."""


