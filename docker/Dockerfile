# vln-annotator — single image for both roles.
#
#   labeling (client)  : python3 annotate_dataset.py ...      -> no GPU needed
#   serving  (server)  : vllm serve nvidia/Cosmos-Reason1-7B  -> needs --gpus all
#
# The annotator itself only needs `openai` + `numpy`; it talks to an
# OpenAI-compatible endpoint over HTTP and never loads a model in-process.
# We nonetheless base on the vLLM image so that one image can also *serve*
# Cosmos Reason locally. See docker/README.md for the tradeoff.
FROM vllm/vllm-openai:v0.28.0

WORKDIR /app

# Only code goes into the image. Data, models and secrets arrive at run time
# via bind mounts and --env-file.
COPY pyproject.toml README.md ./
COPY vln_annotator ./vln_annotator
RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir pillow

COPY annotate_dataset.py ./

ENV HF_HOME=/hf \
    PYTHONUNBUFFERED=1

# The base image pins ENTRYPOINT to `vllm serve`, which would make it
# impossible to run the annotator. Clear it; run.sh / serve.sh each pick
# their own command.
ENTRYPOINT []
CMD ["python3", "annotate_dataset.py", "--help"]
