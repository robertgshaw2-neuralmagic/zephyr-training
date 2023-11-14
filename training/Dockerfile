FROM mosaicml/llm-foundry:2.1.0_cu121_flash2-latest

RUN pip install jupyter trl

RUN git clone https://github.com/mosaicml/llm-foundry.git

RUN pip install -e /llm-foundry

CMD ["/bin/bash"]
