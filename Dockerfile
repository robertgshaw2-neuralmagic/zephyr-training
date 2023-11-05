FROM mosaicml/llm-foundry:2.1.0_cu121_flash2-latest

RUN pip install jupyter trl

CMD ["/bin/bash"]
