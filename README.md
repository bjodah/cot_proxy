# cot_proxy: Intercept and modify requests and responses from OpenAI v1/ endpoints
This is a heavily modified fork of [cot_proxy](https://github.com/bold84/cot_proxy)

## Configuration
See the self-explanatory [examples/cot_proxy.yaml](examples/cot_proxy.yaml).

### Environment Variables
The following variables can be set:

- `COT_TARGET_BASE_URL`: Target API endpoint (default in example: `http://your-model-server:8080/`)
- `COT_CONFIG`: Path to your yaml config file

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
