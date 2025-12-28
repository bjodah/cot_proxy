# Refactoring and feature development
The goal is to move away from configuring the app using environment variables such as:
- THINK_TAG
- THINK_END_TAG
- DEBUG
- API_REQUEST_TIMEOUT
- LLM_PARAMS

with exceptions for:
- TARGET_BASE_URL: rename to `COT_TARGET_BASE_URL`
- COT_CONFIG: new variable, it should contain the path to the new yaml config file.

and instead rely mostly on a YAML config file (see examples/cot_proxy.yaml)

Currently requests are 
