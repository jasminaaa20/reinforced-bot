# Running the Code

## If you have `uv` installed

1. Install dependencies:

    ```bash
    uv sync
    ```

2. Run the application:

    ```bash
    uv run main.py
    ```

## If you don't have `uv`

1. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS/Linux:

      ```bash
      source venv/bin/activate
      ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
    python main.py
    ```
