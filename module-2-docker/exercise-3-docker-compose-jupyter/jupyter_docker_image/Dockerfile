FROM debian:10.12-slim as builder

# download and install miniconda and su-exec
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates build-essential git wget && apt-get clean && apt-get autoremove && \
      rm -rf /var/lib/apt/lists/* && \
      wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
      bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/share/miniconda && \
      rm -rf Miniconda3-latest-Linux-x86_64.sh  && \
      git clone https://github.com/ncopa/su-exec.git && \
      cd su-exec && make su-exec && mv su-exec /usr/local/sbin && \
      rm -rf /home/su-exec

FROM debian:10.12-slim as main
COPY --from=builder /usr/local/share/miniconda /usr/local/share/miniconda
COPY --from=builder /usr/local/sbin/su-exec /usr/local/sbin/su-exec

ENV PATH=/usr/local/share/miniconda/bin:${PATH}

WORKDIR /home

COPY env.yaml ./

RUN conda env create -f env.yaml -n env && rm env.yaml

ENV PATH="/usr/local/share/miniconda/envs/env/bin:${PATH}"
COPY jupyter-settings/ jupyter-settings/
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh 
ENTRYPOINT ["./entrypoint.sh"]

CMD ["python", "-m", "jupyterlab", "--no-browser", "--ip", "0.0.0.0", \
	"--port", "8888", "--ServerApp.token=", \
	"--notebook-dir", "/mnt/host-files", "--ServerApp.password", "${JUPYTER_PASS_HASH}"]
