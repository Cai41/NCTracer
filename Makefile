# Makefile for MapReduce Page Rank project.

# Customize these paths for your environment.
# -----------------------------------------------------------
jar.name=nctracer_2.11-0.1.jar
jar.path=target/scala-2.11/${jar.name}
training.job.name=Training
prediction.job.name=Prediction
local.input=input
local.output=output
local.log=log

# AWS EMR Execution
aws.emr.release=emr-5.8.0
aws.region=us-east-1
aws.bucket.name=cs6240-yang-cai
aws.subnet.id=subnet-9b31f4a4
aws.input=input
aws.output=output
aws.log.dir=log
# number of workers
aws.num.nodes=18
aws.instance.type=m4.large
# -----------------------------------------------------------

# Compiles code and builds jar (with dependencies).
jar:
	sbt package

# Removes local output directory.
clean-local-training-output:
	rm -rf ${local.output}/models

# Remove prediction result
clean-local-prediction-output:
	rm -rf ${local.output}/prediction_result

# Remove local log directory.
clean-local-log:
	rm -rf ${local.log}*

# Runs standalone
alone: jar clean-local-training-output
	spark-submit --class ${training.job.name} --master local[*] --conf "spark.driver.memory=5g" --conf "spark.executor.cores=4" ${jar.path} ${local.input} ${local.output}

alone-prediction: jar clean-local-prediction-output
	spark-submit --class ${prediction.job.name} --master local[*] --conf "spark.driver.memory=5g" --conf "spark.executor.cores=4" ${jar.path} ${local.input} ${local.output}

# Create S3 bucket.
make-bucket:
	aws s3 mb s3://${aws.bucket.name}

# Upload data to S3 input dir.
upload-input-aws: make-bucket
	aws s3 sync ${local.input} s3://${aws.bucket.name}/${aws.input}

# Delete S3 output dir.
delete-output-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output}*"

# Upload application to S3 bucket.
upload-app-aws:
	aws s3 cp ${jar.path} s3://${aws.bucket.name}

# OldBz2WikiParser EMR launch.
cloud: jar upload-app-aws delete-output-aws
	aws emr create-cluster \
		--name "NCTracer 6g lr 200" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
			--applications Name=Spark \
			--steps '[{"Args":["spark-submit","--deploy-mode","cluster","--class","${training.job.name}","--master","yarn","--conf","spark.executor.cores=4","--conf","spark.executor.memory=5g","--conf", "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2","--conf", "spark.serializer=org.apache.spark.serializer.KryoSerializer", "--conf", "spark.task.cpus=2", "s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.output}"],"Type":"CUSTOM_JAR","Jar":"command-runner.jar","ActionOnFailure":"TERMINATE_CLUSTER","Name":"Custom JAR"}]' \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--configurations '[{"Classification":"spark", "Properties":{"maximizeResourceAllocation": "true"}}]' \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate

# Prediction
cloud-prediction: jar upload-app-aws delete-output-aws
	 aws emr create-cluster \
		--name "LR, BY, RF 15worker Prediction" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		 --applications Name=Spark \
		 --steps '[{"Args":["spark-submit","--deploy-mode","cluster","--class","${prediction.job.name}","--master","yarn","--conf","spark.executor.cores=4","--conf","spark.executor.memory=5g","--conf", "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2","--conf", "spark.serializer=org.apache.spark.serializer.KryoSerializer", "--conf", "spark.task.cpus=2", "s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.output}"],"Type":"CUSTOM_JAR","Jar":"command-runner.jar","ActionOnFailure":"TERMINATE_CLUSTER","Name":"Custom JAR"}]' \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--configurations '[{"Classification":"spark", "Properties":{"maximizeResourceAllocation": "true"}}]' \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate

# Download output from S3.
download-output-aws: clean-local-output
	mkdir ${local.output}
	aws s3 sync s3://${aws.bucket.name}/${aws.output} ${local.output}

# Download log from S3.
download-log-aws: clean-local-log
	mkdir ${local.log}
	aws s3 sync s3://${aws.bucket.name}/${aws.log.dir} ${local.log}
