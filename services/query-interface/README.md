# Complex Querying Interface Documentation

This includes the script to launch the interface for complex querying.

## Instructions

1. Run the querying server. Make sure to use a free port number,
   e.g. if you are using Octopus on `8000` then use other port like `8088`:

   ```shell
   SPAICE_MODEL_NAME=querying uv run pythons service.py --port 8088
   ```

   The value you set for `SPAICE_MODEL_NAME` will serve as the base path of all the routes.

2. You can test a query using the sample data. Make sure to use the correct port (e.g. `8088`) and base path (e.g. `querying`) you just opened.
   ```shell
   curl -X POST "http://127.0.0.1:8088/querying/query" -H "Content-Type: application/json" -d samples/request.json"
   ```
