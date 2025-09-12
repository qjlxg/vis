name: Test China C-Class Fund Screener

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install --upgrade akshare
        pip install -r requirements.txt

    - name: Run fund screener
      run: |
        echo "Starting fund screener script..."
        python fund_screener_cn_cclass.py 2>&1 | tee output.log
        echo "Script execution finished. Displaying output log:"
        cat output.log
