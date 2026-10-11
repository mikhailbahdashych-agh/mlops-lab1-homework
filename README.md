# Laboratory 1 homework 

Homework will be implementing a production-ready sentiment analysis application.
For the purposes of this assignment, we focus only on a single prediction, i.e. request-response
for a single text at a time.

Instructions:
- do tasks in order, they have been designed to build upon prior steps
- write clean code, descriptive variable and function names, use comments where necessary etc.
- send your resulting application code either as .zip file or GitHub public repository link

## 1. Dependency management setup (2 points)

- Prepare a development environment with `uv` ([uv commands](https://docs.astral.sh/uv/getting-started/features/)) that contains:
    - `pyproject.toml`
    - `uv.lock`
- To run the inference mentioned below, you need:
    - `scikit-learn`
    - `clean-text`
    - `joblib`
    - `torch`
    - `sentence_transformers`
- To finish all the tasks, you will also need further dependencies. Add them as necessary. All are mentioned
  in the lab instruction, in code, or in the docstring of prepared `inference.py` script.
- Use [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies)
  to separate development dependencies in your `pyproject.toml`. For example, we do not need `pytest` to
  run our inference app on the production server.
- Set up [PyTorch in uv](https://docs.astral.sh/uv/guides/integration/pytorch/) for either CPU or GPU,
  whatever is available to you.

## 2. Pre-commit hooks (1 point)

Set up the pre-commit hooks:
- `ruff` (linter and formatter)
- `mypy`

If you need additional configuration, adding them to `pyproject.toml` is recommended. However, if you prefer,
you can also use separate files for this.

## 3. Webserver development (2 points)

Implement FastAPI webserver for inference, which should have:
- a single endpoint `/predict`, accepting POST requests
- Pydantic model validating request  (input), JSON with single key `text` (string)
- Pydantic model validating response (output), JSON with single key `prediction` (string)
- Fow now, before you stard next point, you can return always 'positive'

To test the application, you can use [Swagger UI](https://fastapi.tiangolo.com/#interactive-api-docs)
(provided by FastAPI), `curl`, `requests` or any other library. `curl` code would be:

```bash
curl -X POST \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "What a great MLOps lecture, I am very satisfied"}'
```

The sample expected response is: 
```json 
{ "prediction": "positive" } 
``` 

## 4. Sentiment analysis implementation (2 points)

Your teammate, the data scientist, has already prepared the sentiment analysis model for you: logistic
regression on top of Sentence Transformer embeddings. Transformer was saved with `sentence-transformers`
library, and logistic regression with `joblib` (for an overview of alternatives, see [scikit-learn docs](https://scikit-learn.org/stable/model_persistence.html)).
He shared it with Google Drive: https://drive.google.com/file/d/1NRZdYq5jweVRUzAZG518LMhs4E56IgxG/view?usp=share_link

Download and unpack the model. It contains two models, that you need to load and integrate.
See `sentence-transformers` and scikit-learn docs for model loading instructions.

Text goes into the Sentence Transformer, embedding goes out, and is used as an input to the
logistic regression. It returns one of 3 classes for text sentiment: 0 (negative), 1 (neutral), or 2 (positive).

Add the model to your FastAPI server. Return predicted class as a string.

## 5. Code testing (1 point)

Implement unit tests with `pytest`:
- test input validation, model inference, and response validation
- input text should be a non-empty string
- model should be loaded from provided cloudpickle file without errors
- inference should work for a few sample strings
- output should be a valid JSON response
- for an invalid input, validation should return a valid JSON with error explanation

## 6. Containerization (2 points)

Implement Docker and Docker Compose to containerize your application, so it:
- is packaged in a Docker container, with code and model
- contains all required inference dependencies
- use official uv Docker image 
- create `.dockerignore` file to exclude `.venv/`, `__pycache__/`, etc
- use cache mounts (`--mount=type=cache`) for faster rebuilds
- use bind mounts (`--mount=type=bind`) for dependency files
- is runnable using `docker compose up`
- is accessible on `http://localhost:8000`

## Grading:
- section 1: 2 points
- section 2: 1 point
- section 3: 2 points
- section 4: 2 points
- section 5: 1 point
- section 6: 2 points