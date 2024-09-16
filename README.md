# Spark-Demo
Container Image Setup 

(This step is the same as above) Download Apache Spark - I used version 3.3.2

Apache Download Mirrors - or visit the downloads page here Downloads | Apache Spark 

untar the file into a directory called spark 

Downloaded the rapids accelerator for apache spark: Download - I downloaded version 23.06.0

Download the GPU discovery script 

wget https://raw.githubusercontent.com/apache/spark/master/examples/src/main/scripts/getGpusResources.sh

Download the dockerfile needed to build the image

Downloaded dockerfile.cuda https://nvidia.github.io/spark-rapids/docs/get-started/Dockerfile.cuda 

In your directory you should have:

the .jar file from step 2

the GPU discover script from step 3

dockerfile.cuda from step 4

the folder of the untarred content from step 1 

Modify the dockerfile.cuda file to change the image that it is built FROM, the image that the dockerfile comes with does not exist anymore; additionally add the ARG DEBIAN_FRONTEND=noninteractive so you aren't prompted for any information as the container is building


Generic Command:
docker buildx build . --platform linux/amd64 --push -f Dockerfile.cuda -t $IMAGE_NAME

Example Command: 
docker buildx build . --platform linux/amd64 --push -f Dockerfile.cuda -t vivekkolasani1996/spark:3.3


How to run in RUN.AI

create service account:-

Generic Command:
kubectl create serviceaccount spark -n <runai-project-namespace>

Example Command:
kubectl create serviceaccount spark -n runai-spark-demo


clusterrolebinding to give the spark workload the permissions to run

Generic Command:
kubectl create clusterrolebinding spark-role --clusterrole edit --serviceaccount <runai-project-namespace>:spark -n <runai-project-namespace>

Example Command:
kubectl create clusterrolebinding spark-role --clusterrole edit --serviceaccount runai-spark-demo:spark -n runai-spark-demo


./spark/bin/spark-submit --master k8s://https://csit-spark.runailabs-cs.com:6443 \
--deploy-mode cluster \
--name spark-rapids-team-a \
--class org.apache.spark.examples.SparkPi \
--conf spark.executor.instances=3 \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.executor.memory=4G \
--conf spark.executor.cores=1 \
--conf spark.task.cpus=1 \
--conf spark.task.resource.gpu.amount=1 \
--conf spark.rapids.memory.pinnedPool.size=2G \
--conf spark.executor.memoryOverhead=3G \
--conf spark.sql.files.maxPartitionBytes=512m \
--conf spark.sql.shuffle.partitions=10 \
--conf spark.plugins=com.nvidia.spark.SQLPlugin \
--conf spark.kubernetes.namespace=runai-spark-demo \
--conf spark.kubernetes.driver.pod.name=spark-rapids-driver \
--conf spark.executor.resource.gpu.discoveryScript=/opt/sparkRapidsPlugin/getGpusResources.sh \
--conf spark.executor.resource.gpu.vendor=nvidia.com \
--conf spark.kubernetes.container.image=robmagno/spark-rapids:3.3 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--conf spark.executor.extraClassPath=/opt/sparkRapidsPlugin/rapids-4-spark_2.12-23.06.0.jar \
--conf spark.driver.extraClassPath=/opt/sparkRapidsPlugin/rapids-4-spark_2.12-23.06.0.jar \
--conf spark.kubernetes.scheduler.name=runai-scheduler \
--conf spark.kubernetes.driver.label.runai/queue=spark-demo \
--conf spark.kubernetes.executor.label.runai/queue=spark-demo \
--driver-memory 6G \
local:///opt/spark/examples/jars/spark-examples_2.12-3.3.2.jar 1000000
