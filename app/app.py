from flask import Flask, render_template, request, jsonify
import obtainFinancials
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    # Extract data from the request
    data = request.json
    
    # Call the Python script to process the data
    result = obtainFinancials.process_data(data)
    
    # Return the result as JSON
    return jsonify(result)

# Define API routes for interacting with the database
# Example:
@app.route('/api/add_data', methods=['POST'])
def add_data_to_database():
    # Extract data from the request
    data = request.json
    
    # Add data to the database (implementation depends on your database setup)
    # Example: database.add(data)
    
    return jsonify({"message": "Data added to the database"})

if __name__ == '__main__':
    app.run(debug=True)
