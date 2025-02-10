import requests

url = "http://127.0.0.1:5000/endpoint"
payload = {
    "text": "Project Phoenix: A Comprehensive Plan to Expand Redwood Tech Solutions\n\n### Table of Contents\n1. [Introduction](#introduction)\n2. [Project Objectives](#project-objectives)\n3. [Key Stakeholders](#key-stakeholders)\n4. [Market Analysis](#market-analysis)\n5. [Product Roadmap](#product-roadmap)\n6. [Budget & Resource Allocation](#budget--resource-allocation)\n7"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
