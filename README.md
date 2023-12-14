<div align="center">

# Face Verification API

</div>

### Cloning the repository

--> Clone the repository using command below :

```bash
git clone https://https://github.com/heptavators/friendease-ml-face-verification.git
```

--> Move into the directory :

```bash
cd friendease-ml-face-verification
```

--> Create a virtual environment :

```bash
# Install virtualenv first
pip install virtualenv

# Then create virtual environment
virtualenv venv
```

--> Activate the virtual environment :

```bash
venv\Scripts\activate
```

--> Install the requirements :

```bash
pip install -r requirements.txt
```

--> Run the code :

```bash
uvicorn src.api.main:app --reload
```
