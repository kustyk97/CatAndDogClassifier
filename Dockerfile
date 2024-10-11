# Użyj obrazu bazowego PyTorch
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Ustaw zmienną środowiskową dla użytkownika
ENV USER jovyan

# Zainstaluj Jupyter Notebook i inne potrzebne pakiety
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install --no-cache-dir jupyter && \
    pip install matplotlib && \
    pip install seaborn

# Ustaw port, na którym będzie działał Jupyter Notebook
EXPOSE 8888

CMD ["/bin/bash"]
# Uruchom Jupyter Notebook przy starcie kontenera
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]