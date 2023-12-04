# LocalStack Doc

- create a lambda function

```commandline
awslocal lambda create-function \
--function-name functionName \
--runtime python3.9 \
--zip-file fileb://function.zip \
--handler lambda_handler \
--role arn:aws:iam::000000000000:role/lambda-role
```

- invoke a function

```commandline
awslocal lambda invoke --function-name localstack-lambda-url-example \
    --payload '{"body": "{\"num1\": \"10\", \"num2\": \"10\"}" }' output.txt
```