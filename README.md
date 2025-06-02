bomare_sav_chatbot

# SAV Bomare Chatbot

This project is a web application with a React frontend, Flask backend, and Supabase database.


## Setup Guide

Follow the steps below to set up and run the project locally after cloning the repository.


### Prerequisites

- [Node.js and npm](https://nodejs.org/en/download/)
- [Python 3](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)


### Clone the repository

```bash
git clone https://github.com/imenedjelili/bomare_sav_chatbot.git
cd bomare_sav_chatbot
````

---

## Frontend (React)

1. Change directory to the frontend_ without_node_modules folder:

```bash
cd 'frontend_ without_node_modules'
```

2. Install the dependencies:

```bash
npm install
```

3. Start the React development server:

```bash
npm start
```

Your React app should now be accessible at [http://localhost:3000](http://localhost:3000).

---

## Backend (Flask)

1. Change directory to the backend folder:

```bash
cd backend
```

2. Create a Python virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

* On macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

* On Windows (Command Prompt):

  ```bash
  venv\Scripts\activate
  ```

4. Install the required Python packages:

```bash
pip install -r requirements.txt
```

5. Run the Flask backend server:

```bash
python app.py
```
or
```bash
flask run
```

The backend server will run (usually on [http://localhost:5000](http://localhost:5000)).

---

## Database (Supabase)

Make sure to set up your Supabase database and update the backend configuration with your Supabase URL and API keys as environment variables or in your config files.

---

## Important Notes

* **Do not commit** `node_modules` or `venv` directories to GitHub.
* When adding new dependencies:

  * For the frontend, run:

    ```bash
    npm install <package-name> --save
    ```

    This updates `package.json`.

  * For the backend, after installing packages, update `requirements.txt`:

    ```bash
    pip freeze > requirements.txt
    ```

---

## Troubleshooting

* If you have trouble activating the virtual environment:

  * On Windows, try running the terminal as Administrator.
  * On macOS/Linux, make sure your shell allows sourcing scripts.

* Make sure you have compatible versions of Node.js and Python installed.



```

```
