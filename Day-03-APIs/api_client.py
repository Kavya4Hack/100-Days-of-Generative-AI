"""
Day 03 — APIs
Practice HTTP requests, JSON responses, headers and environment variables.

This example uses a public test endpoint. Replace it with a real API only when
you have a valid API key and permission to use the service.
"""

import os
import requests


def get_demo_post(post_id=1):
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def post_demo_data(title, body):
    url = "https://jsonplaceholder.typicode.com/posts"
    payload = {"title": title, "body": body, "userId": 1}

    response = requests.post(
        url,
        json=payload,
        headers={"Accept": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print(get_demo_post())
    print(post_demo_data("API Practice", "Learning HTTP and JSON."))
    print("Example API key loaded:", bool(os.getenv("API_KEY")))
