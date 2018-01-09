FROM demikandr/pytorch


COPY . . 
RUN pip install -q tqdm
# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output
# WORKDIR /submission
RUN ls
CMD ["python", "Evaluate.py"]
