FROM mosaicml/llm-foundry:2.1.0_cu121_flash2-latest

RUN pip install jupyter trl

COPY llm-foundry /llm-foundry

RUN pip install -e /llm-foundry

CMD ["/bin/bash"]
